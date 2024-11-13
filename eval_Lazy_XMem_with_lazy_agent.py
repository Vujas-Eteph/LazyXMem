"""
Lazy-XMem: Lazy Video Object Segmentation/Tracking
"""

# - IMPORTS ---
# Standard library imports
import os
from os import path

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from progressbar import progressbar
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

# Handling optional imports
try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')

import colorful as cf
from icecream import ic
import lovely_tensors as lt
lt.monkey_patch()

# Local application/library specific imports
from utils.entropy_module import EntropyHelper, store_entropy_value_and_compute_derivative_per_object, get_iou_between_pd_and_gt_obx_mask, get_IoU
from inference.data.mask_mapper import MaskMapper
from inference.destructured_inference_core import DestructuredInferenceCore
from model.network import XMem

# Custom packages
from ivots_robot import ivots_robot
from sam_hq_wrapper import Mask_Refiner, DebugSAM, refine_prediction_with_SAM

# - Arguments ---
from utils.params_passer import args_passer

# - FUNCTIONS ---
from utils.distance_map_module import generate_negative_coords
from utils.meta_dataset_handler import create_meta_dataset
from utils.translator_for_SAM_and_XMem import prep_transformation_and_image_for_SAM_and_XMem
from utils.data_transformations import make_a_flip, data_info, resize_entropy_map_for_SAM, adapt_img_gt_mask_pd_mask_for_SAM
from utils.prompt_functions import init_prompt_tracker, add_new_obj_to_prompt_tracker, add_new_entry_to_prompt_tracker, call_a_prompt_for_obx, adapt_prompts_for_the_first_frame, record_entropy_values, get_prompt_for_sam, find_pseudo_interaction_coords, find_user_interaction_coords
from utils.curve_recorders import FrameLevelCurveRecorder, UpdateMemoryFlag
from utils.debug_functions import debug_show_entropy, debug_phase_2, debug_phase_3

args = args_passer()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']

# Classes
entropy_crafter = EntropyHelper()
zivos_robot = ivots_robot.IvotsRobot()
mask_refiner = Mask_Refiner(debug_mode=args.debug,
                            use_save_feature=args.debug)
if args.debug:
    SAM_debugger = DebugSAM(use_save_feature=True)

if args.verbose:
    ic.disable()

args.output = f"{args.output}_IoUat{args.iouatX}_Hat{args.HatX}_Deri{args.derivatX}_Temperature_{args.temperature}_working_{args.working_upd}_deep_{args.deep_upd}_mem_upd_{args.Mem_upd}"

if args.output is None:
    args.output = f'../output/{args.dataset}'
    print(f'Output path not provided. Defaulting to {args.output}')

with open(os.path.join('conf', 'eval_config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

dataset_name = args.dataset
data_cfg = cfg.get('datasets').get(dataset_name)
is_burst = ('burst' in dataset_name)
# setup dataset
image_dir = data_cfg.get('image_directory')
meta_dataset = create_meta_dataset(data_cfg, is_burst, image_dir)

print(cf.bold_white(cf.red("WARNING")), "We have",
      cf.bold_white(cf.orange("use_all_masks=True")), "in the dataloader")

torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()

# Load our checkpoint
network = XMem(config, args.model).cuda().eval()
if args.model is not None:
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
else:
    print('No model loaded.')

# Start eval
prompt_meta_data = dict()
entropy_seuil = args.HatX  # if above 0.5, use the GT mask
prompt_meta_data["Threshold"] = entropy_seuil
prompt_meta_data["IoU@"] = args.iouatX
prompt_meta_data["Derivative"] = args.derivatX


def predict_with_SAM_using_clicks(mask_refiner, input_coords, input_labels)-> np.ndarray:
    mask_predicted, score, logits = mask_refiner.refine_mask(point_coords=input_coords,point_labels=input_labels,
                                                             multimask_output=True, hq_token_only=True)
    return mask_predicted

def show_mask(mask, ax, random_color=False, color=[30,144,255]):
    color_transprante = np.array([color[0]/255, color[1]/255, color[2]/255, 0.6])
    color_hard = np.array([color[0]/255, color[1]/255, color[2]/255, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color_transprante.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # Draw black contour around the mask
    ax.contour(mask, levels=[0.5], colors=[color_hard], linewidths=2)  
    

def show_points(coords, ax, marker_size=350):
    if coords[2] == 1:
        ax.scatter(coords[0], coords[1], color='yellow', marker='*', s=marker_size, edgecolor='black', linewidth=1.5)
    if coords[2] == 0:
        ax.scatter(coords[0], coords[1], color='red', marker='*', s=marker_size, edgecolor='black', linewidth=1.5)


def show_res(image, masks, input_point, iou_list):
    fig, ax = plt.subplots(2, int(len(image)/2) ,figsize=(20,20))
    idx = 0
    for i in range(0,2):
        for j in range(0,int(len(image)/2)):
            img, msk, pnt, iou =  image[idx], masks[idx], input_point[idx], iou_list[idx]
            ax[i,j].imshow(img)
            show_mask(msk, ax[i,j])
            show_points(pnt, ax[i,j])
            idx = idx + 1

    plt.axis('off')
    plt.savefig('Show_IMAGE.png', bbox_inches='tight', pad_inches=0, dpi=300)


def reconstruct_mask_for_SAM(image, zivos_robot, mask_refiner, mask:torch.Tensor):
    # Generate the central interaction
    msk = mask.detach().cpu().numpy()
    iou_score = 0
    history_of_interactions = []
    history_of_iou = []
    history_of_masks = []
    history_of_logits = []
    history_of_images = []
    idx = 0
    while (iou_score < 0.8) and (idx < 8):
        if idx == 0:
            robot_click_coords = zivos_robot.get_interaction(msk, msk, mode="TP")
            input_coords = np.array([[robot_click_coords['w'], robot_click_coords['h']]])
            input_labels = np.array([1])
            new_mask, scores, logits = mask_refiner.refine_mask_using_a_base_SAM_mask(None, input_coords, input_labels)
            new_mask = new_mask[np.argmax(scores)]
            mask_input = logits[np.argmax(scores), :, :][None,:,:]
        else:
            robot_click_coords = zivos_robot.get_interaction(new_mask, msk, mode="All")
            input_coords = np.array([[robot_click_coords['w'], robot_click_coords['h']]])
            if 'FP' == robot_click_coords['Region']:
                input_labels = np.array([0])
            if 'FN' == robot_click_coords['Region']:
                input_labels = np.array([1])
            new_mask, _, mask_input = mask_refiner.refine_mask_using_a_base_SAM_mask(mask_input, input_coords, input_labels, multimask_output=False)
            new_mask = new_mask[0]
        idx = idx + 1
        iou_score = get_IoU(new_mask, msk)
        history_of_iou.append(iou_score)
        history_of_interactions.append([robot_click_coords['w'], robot_click_coords['h'], input_labels.item()])
        history_of_masks.append(new_mask)
        history_of_images.append(image)
        history_of_logits.append(mask_input)
        
    # show_res(history_of_images, history_of_masks, history_of_interactions, history_of_iou)
    
    best_index = np.argmax(history_of_iou).item()
    return history_of_masks[best_index], history_of_logits[best_index]


def init_subsitute_masks(init_labels_list, out_mask_og, substitute_mask):
    for e in init_labels_list:
        one_hot_mask = e == out_mask_og
        substitute_mask[e-1, :, :] = torch.from_numpy(one_hot_mask)


def select_relevant_mask(mask, obx):
    return torch.from_numpy(mask == obx)

for vdx, vid_reader in enumerate(progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True)):
    loader = DataLoader(vid_reader,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers)
    
    vid_name = vid_reader.vid_name    
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term'] and
        (vid_length
            / (config['max_mid_term_frames']-config['min_mid_term_frames'])
            * config['num_prototypes'])
        >= config['max_long_term_elements']
    )

    mapper = MaskMapper()
    processor = DestructuredInferenceCore(network, config=config)
    
    prompts_seq_dict = dict()
    entropy_memory = dict()
    first_mask_loaded = False
    for fdx, data in enumerate(loader):
        with (torch.cuda.amp.autocast(enabled=not args.benchmark)):
            rgb, rgb_raw, msk, first_msk, info, frame, img_shape, need_resize = data_info(data)

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # if, for some reason, the first frame is not aligned with the first mask
            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            rgb, msk, first_msk = make_a_flip(args, rgb, msk, first_msk)

            # Map possibly non-continuous labels to continuous ones
            if first_msk is not None:
                # In this region, add the missing frame for the first problem 
                # sequence
                msk, labels = mapper.convert_mask(first_msk[0].numpy(),
                                                  exhaustive=True)
                msk = torch.Tensor(msk).cuda()
                if not is_burst:
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
            else:
                if msk is not None:
                    # In this region, add the missing frame for the first problem
                    # sequence
                    msk, labels = mapper.convert_mask(msk[0].numpy(),
                                                      exhaustive=True)
                    msk = torch.Tensor(msk).cuda()
                    if not is_burst:
                        if need_resize:
                            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                else:
                    labels = None

            GT_labels = labels
            GT_msk = msk
            if fdx == 0:
                init_labels_list = [l for l in labels]
                frame_0_labels = init_labels_list
                prompts_frame_dict = {}
                init_prompt_tracker(init_labels_list, prompts_frame_dict)

                resize_from_sVOS_baseline_to_SAM, resize_from_sVOS_baseline_to_SAM_bilinear_variant, resize_from_SAM_to_sVOS_baseline = prep_transformation_and_image_for_SAM_and_XMem(rgb, rgb_raw)
                
            if (first_msk is not None) and fdx != 0:
                init_labels_list = [l for l in labels]
                for obx in init_labels_list:
                    # Don't delelte the info from the first elements
                    if obx in frame_0_labels: 
                        continue
                    add_new_obj_to_prompt_tracker(prompts_frame_dict, obx)
                frame_0_labels = init_labels_list

            if fdx != 0 and first_msk is None:
                msk = None
                labels = None

            XMem_prob = processor.segment_fdx(rgb, msk, labels, end=(fdx == vid_length-1))
            
            shape_for_substitute_XMem_mask = XMem_prob.shape

            out_mask_og = torch.max(XMem_prob, dim=0).indices
            out_mask_og = (out_mask_og.detach().cpu().numpy()).astype(np.uint8)

            if processor.prob_pmf_per_object is not None:
                prob_pmf_per_object = processor.prob_pmf_per_object
            else:
                prob_pmf_per_object = None

            # Upsample to original size if needed
            if need_resize:
                XMem_prob = F.interpolate(XMem_prob.unsqueeze(1), img_shape, mode='bilinear', align_corners=False)[:, 0]
                if processor.prob_pmf_per_object is not None:
                    prob_pmf_per_object = F.interpolate(prob_pmf_per_object.unsqueeze(1), img_shape,
                                                        mode='bilinear', align_corners=False)[:,0]
            if args.flip:
                XMem_prob = torch.flip(XMem_prob, dims=[-1])
                if processor.prob_pmf_per_object is not None:
                    prob_pmf_per_object = torch.flip(prob_pmf_per_object, dims=[-1])
                    
            if processor.prob_pmf_per_object is not None:
                prob_pmf_per_object = prob_pmf_per_object.detach().cpu().numpy()

            XMem_probability_map = XMem_prob.clone().detach().cpu().numpy()
            XMem_mask = torch.max(XMem_prob, dim=0).indices
            XMem_mask = (XMem_mask.detach().cpu().numpy()).astype(np.uint8)
            
            # Initiate instanced to record curves 
            fdx_curve_recorder = FrameLevelCurveRecorder()
            update_memory = UpdateMemoryFlag()
            
            percentage = 5
            
            # Compute entropy Map       
            normalized_entropy_array_edition, normalized_entropy_tensor_edition = entropy_crafter.compute_normalized_entropy(XMem_prob.clone().detach().cpu().numpy())

            # Compute the maksed entropy
            for obx in init_labels_list:
                fdx_curve_recorder.entropy_memory_frame_map[obx], fdx_curve_recorder.entropy_memory_frame[obx] = entropy_crafter.get_masked_entropy_for_obx(XMem_mask, obx, normalized_entropy_array_edition, percentage)
                entropy_deriv_odx = store_entropy_value_and_compute_derivative_per_object(entropy_memory, fdx, obx, fdx_curve_recorder.entropy_memory_frame)
                
                # Considere the IoU or other similar metric to issue a prompt
                obx_gt_mask, obx_IoU = get_iou_between_pd_and_gt_obx_mask(img_shape, need_resize, GT_msk, obx, XMem_mask)
                
                call_a_prompt_for_obx(obx, args, entropy_seuil, fdx_curve_recorder, obx_IoU, entropy_deriv_odx)
                add_new_entry_to_prompt_tracker(prompts_seq_dict, fdx, prompts_frame_dict, obx, fdx_curve_recorder.entropy_memory_frame, obx_IoU, fdx_curve_recorder.call_user_inter_for_obx[-1])
                
                # Update the memory based on the entropy (standard update)
                if args.Mem_upd < fdx_curve_recorder.entropy_memory_frame[obx]:
                    update_memory.flag = False

                if args.debug:
                    debug_show_entropy(SAM_debugger, vid_name, fdx, rgb_raw, obx, normalized_entropy_array_edition, XMem_mask, fdx_curve_recorder.entropy_memory_frame_map, obx_gt_mask, obx_IoU)

            if fdx != 0:
                record_entropy_values(entropy_memory, fdx, fdx_curve_recorder.entropy_memory_frame)
            if fdx == 0:
                adapt_prompts_for_the_first_frame(fdx_curve_recorder.call_user_inter_for_obx, fdx_curve_recorder.call_pseudo_inter_for_obx)
                

            # -----------------------------
            # - Dealing with Pseudo Prompts
            if max(fdx_curve_recorder.call_pseudo_inter_for_obx):
                substitute_mask = torch.ones([len(init_labels_list), *shape_for_substitute_XMem_mask[1:]])
                # Add the mask from prediction to the subsitute_mask
                init_subsitute_masks(init_labels_list, out_mask_og, substitute_mask)
                    
                for obx_pseudo, need_pseudo_prompt in enumerate(fdx_curve_recorder.call_pseudo_inter_for_obx):
                    if not need_pseudo_prompt:
                        continue
                    update_memory.flag = args.update_mem_with_pseudo_prompt  # usually set to False
                    _Xmem_mask_obx_tensor = select_relevant_mask(XMem_mask, obx_pseudo + 1)
                    
                    # Read the confidence map:
                    _masked_entropy_map_tensor_SAM = resize_entropy_map_for_SAM(resize_from_sVOS_baseline_to_SAM_bilinear_variant, fdx_curve_recorder, obx_pseudo)
                    _Xmem_mask_obx_tensor_SAM = resize_from_sVOS_baseline_to_SAM(_Xmem_mask_obx_tensor[None,None,:,:])[0,0]
                    
                    # Here we use the robot to generate an interaction based on the confidence map
                    robot_click_coords = find_pseudo_interaction_coords(fdx, args, zivos_robot, init_labels_list, _Xmem_mask_obx_tensor_SAM, _masked_entropy_map_tensor_SAM, args.debug)
                    
                    # Refine the mask with this pseudo interaction
                    input_coords, input_labels, BBox = get_prompt_for_sam(args, robot_click_coords)                    
                    
                    rgb_raw_for_refinement = resize_from_sVOS_baseline_to_SAM(rgb_raw.permute(0,3,1,2)).permute(0,2,3,1).detach().cpu().numpy()[0]
                    
                    mask_refiner.set_image_for_SAM(rgb_raw_for_refinement)
                    
                    if args.using_mask_in_pseudo:
                        # Similarly to EVA-VOS reconstruct the original mask with a certain number of interactions. This did not yield better results in my case
                        reconstructed_mask_from_SAM, mask_input = reconstruct_mask_for_SAM(rgb_raw_for_refinement, zivos_robot, mask_refiner, _Xmem_mask_obx_tensor_SAM)
                        SAM_pred_mask, _, _ = mask_refiner.refine_mask_using_a_base_SAM_mask(mask_input, input_coords, input_labels, multimask_output=False)
                        
                        new_mask = resize_from_SAM_to_sVOS_baseline(torch.from_numpy(SAM_pred_mask)[None,:,:,:]).type(substitute_mask.dtype)
                        substitute_mask[obx_pseudo, :, :] = new_mask[0]
                        
                    else:
                        mask_refiner.set_mask_for_SAM()
                        refine_prediction_with_SAM(mask_refiner, resize_from_SAM_to_sVOS_baseline, substitute_mask, obx_pseudo, input_coords, input_labels, BBox)   

                    if args.debug:
                        normalized_entropy_tensor_edition = debug_phase_2(mask_refiner, SAM_debugger, vid_name, fdx, GT_msk, resize_from_sVOS_baseline_to_SAM, obx, normalized_entropy_tensor_edition, obx_pseudo, _Xmem_mask_obx_tensor_SAM, rgb_raw_for_refinement)

            # --------------------------------
            # - Dealing with user interactions
            if max(fdx_curve_recorder.call_user_inter_for_obx):
                substitute_mask = torch.ones([len(init_labels_list), *shape_for_substitute_XMem_mask[1:]])
                # Add the mask from prediction to the subsitute_mask
                init_subsitute_masks(init_labels_list, out_mask_og, substitute_mask)

                for obx, need_prompt in enumerate(fdx_curve_recorder.call_user_inter_for_obx):
                    # Replace the mask from the prediction with the mask from the GT                    
                    if not need_prompt:
                        continue
                    update_memory.flag = args.update_mem_with_prompt  # usually set to True
                    _Xmem_mask_obx_tensor = select_relevant_mask(XMem_mask, obx + 1)
                    
                    # Resizing some stuff for SAM
                    _Xmem_mask_obx_tensor_SAM, resized_gt_mask_for_irobot, rgb_raw_for_SAM = adapt_img_gt_mask_pd_mask_for_SAM(rgb_raw, GT_msk, resize_from_sVOS_baseline_to_SAM, obx, _Xmem_mask_obx_tensor)
                    
                    # Generate a TP intereaction in the center of the object                
                    robot_click_coords = find_user_interaction_coords(zivos_robot, _Xmem_mask_obx_tensor_SAM.detach().cpu().numpy(), resized_gt_mask_for_irobot.detach().cpu().numpy())
                    # Create Click interaction
                    if robot_click_coords["Region"] in ["FP","TN"]:
                        interaction_value = 0
                    if robot_click_coords["Region"] in ["FN","TP"]:
                        interaction_value = 1
                    input_labels = np.array([interaction_value])
                    
                    # Create BBox interaction
                    input_coords = np.array([[robot_click_coords['w'], robot_click_coords['h']]])

                    # Prep Images for SAM
                    rgb_raw_for_refinement = rgb_raw_for_SAM.detach().cpu().numpy()[0]
                    mask_refiner.set_image_for_SAM(rgb_raw_for_refinement)
                    mask_refiner.set_mask_for_SAM()

                    # Create negative clicks belonging to other objects and to the background
                    if args.use_negative_clicks:
                        neg_input_coords, neg_input_labels = generate_negative_coords(obx, resize_from_sVOS_baseline_to_SAM(torch.from_numpy(XMem_mask)[None,None,:,:])[0].cpu().numpy(), len(init_labels_list))
                        # TODO: Not used atm, but would also include negative clicks...
                        if len(neg_input_labels) != 0:
                            all_input_coords = np.concatenate((input_coords, neg_input_coords), axis=0)
                            all_input_labels = np.concatenate((input_labels, neg_input_labels))
                            input_coords = all_input_coords
                            input_labels = all_input_labels
                    
                    refine_prediction_with_SAM(mask_refiner, resize_from_SAM_to_sVOS_baseline, substitute_mask, obx, input_coords, input_labels)
                    
                    if args.debug:
                        debug_phase_3(SAM_debugger, vid_name, fdx, resize_from_sVOS_baseline_to_SAM, obx, normalized_entropy_tensor_edition, _Xmem_mask_obx_tensor_SAM, rgb_raw_for_refinement, resized_gt_mask_for_irobot)
                        mask_refiner.save_results_under(f"{vid_name}_{fdx}_{obx}_B")

                # Update the model's data with the substitute mask
                prob_sub = processor.update_with_an_input_mask_w_memory_update(substitute_mask, GT_labels,
                                                                               args.working_upd, args.deep_upd)

                # Upsample to original size if needed
                if need_resize:
                    prob_sub = F.interpolate(prob_sub.unsqueeze(1), img_shape,
                                             mode='bilinear', align_corners=False)[:,0]
                if args.flip:
                    prob_sub = torch.flip(prob_sub, dims=[-1])

            # Update the memory
            if update_memory.flag:
                processor.update_memory_state()
            else:
                processor.only_update_hidden_state()

            # Save the mask
            if args.save_all or info['save'][0]:
                out_path = path.join(args.output, dataset_name, 'Annotations')
                this_out_path = path.join(out_path, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                XMem_mask = mapper.remap_index_mask(XMem_mask)
                out_img = Image.fromarray(XMem_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if args.save_scores:
                np_path_softmax = path.join(args.output, dataset_name,
                                            'softmax', vid_name)
                
                os.makedirs(np_path_softmax, exist_ok=True)
                if fdx == len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path_softmax, f'backward.hkl'), mode='w') # What does that do again ??
                if args.save_all or info['save'][0]:
                    hkl.dump(XMem_probability_map, path.join(np_path_softmax, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')
        
    # Save the prompts results
    if args.save_prompts:
        np_path_prompt = path.join(args.output, dataset_name, 'prompts', vid_name)
        os.makedirs(np_path_prompt, exist_ok=True)
        for key, value in prompts_seq_dict.items():
            df = pd.DataFrame(value)
            file_name = os.path.join(np_path_prompt, f"{key}.parquet")
            df.to_parquet(file_name)
        # Convert the json into a polar dataframe and save as parquet file for each object

print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

if args.save_prompts:
    df = pl.DataFrame(prompt_meta_data)
    path_to_save_results = os.path.join(args.output, dataset_name, 'prompts')
    os.makedirs(path_to_save_results, exist_ok=True)
    file_name = os.path.join(path_to_save_results, "meta_data.parquet")
    df.write_parquet(file_name)
