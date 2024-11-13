"""
This evaluation generates prompts avery time the the IoU is lower than the
specified IoU threshold.

Simply applies an interaction on the 25%, 50% and 75% regions. Or every 20th frame "apply an interaction" 

Quick use:
- python3 eval_regulated_with_IoU@X.py --output ../output_for_inference/Single_Models/UXMem_Regulated_with_IoU@X --save_scores
- python3 eval_regulated_with_IoU@X.py --output ../output_for_inference/Single_Models/UXMem_Regulated_with_IoU@X --save_scores --iouatX 0.9
- python3 eval_regulated_with_IoU@X.py --output ../output_for_inference/Single_Models/UXMem_Regulated_with_IoU@X --save_scores --deep_update
- python3 eval_regulated_with_IoU@X.py --output ../output_for_inference/Single_Models/UXMem_Regulated_with_IoU@X --save_scores --iouatX 0.9 --deep_update

python3 eval_iXMem.py --output ../output_for_inference/Single_Models/UXMem_Regulated_no_update_with_IoU@X_weightS_s012 --save_scores --dataset d17-val --model ./saves/XMem.pth 

Modified by Stéphane Vujasinovic
"""

# - IMPORTS ---
import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset, BURSTDetectionTestDataset, MOSETestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.destructured_inference_core import DestructuredInferenceCore

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')


from inference.data.vos_test_dataset import VOSTestDataset
from inference.data.burst_test_dataset import BURSTTestDataset
from inference.data.burst_utils import BURSTResultHandler
from inference.data.args_utils import get_dataset_cfg
import yaml

from icecream import ic
import colorful as cf

from eteph_tools.statistics.entropy_operations import EntropyHelper

import lovely_tensors as lt
lt.monkey_patch()

stat_api = EntropyHelper()
import cv2
import polars as pl

from util.tensor_util import pad_divide_by, unpad

# - FUNCTION ---
def gen_kernel(
    x: int,
    object_size: int
) -> np.ndarray:
    # using sqrt because the object growth is square based
    kernel_size = int(np.sqrt((x / 100) * object_size))
    if kernel_size % 2 == 1:  # if kernel is odd
        kernel_size = kernel_size + 1
    kernel_size = max(2, kernel_size)
    # Create a circle based kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    center_of_kernel = [int(kernel.shape[0] / 2),
                        int(kernel.shape[0] / 2)]
    radius = center_of_kernel[0] - 1
    points_y = np.arange(0, int(center_of_kernel[0]))
    points_x = np.arange(0, int(center_of_kernel[0]))
    points_yy, points_xx = np.meshgrid(points_y, points_x)
    points = np.stack((points_yy.flatten(),
                       points_xx.flatten()), axis=-1)
    distance = np.square(points[:, 0]) + np.square(points[:, 1])
    in_circle = distance < np.square(radius)
    one_fourth_of_the_array = in_circle.reshape(center_of_kernel[0], -1)
    kernel[center_of_kernel[0]:, center_of_kernel[0]:] = \
        one_fourth_of_the_array
    kernel[center_of_kernel[0]:, :center_of_kernel[0]] = \
        one_fourth_of_the_array[:, ::-1]
    kernel[:center_of_kernel[0], :] = kernel[center_of_kernel[0]:, :][::-1]

    return kernel



def get_masked_entropy(
    bool_mask: np.ndarray,
    H_fdx: np.ndarray,
    value_for_mask_H: int,
    debug=False
) -> np.ndarray:
    kernel = gen_kernel(value_for_mask_H, bool_mask.sum())
    # Dilate the mask based on the kernel
    Elem = bool_mask
    Elem = Elem.astype(np.uint8)
    dilated_mask = cv2.dilate(Elem, kernel, iterations=1).astype(bool)
    H_fdx = H_fdx * dilated_mask
    if debug:
        cv2.imshow("dilates_mask", dilated_mask.astype(np.uint8) * 255)
        cv2.imshow("not_dilates_mask", Elem * 255)

        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()

    return H_fdx


def get_IoU(
    pd_bool_mask: np.ndarray,
    gt_bool_mask: np.ndarray
):
    """Compute the IoU between the predicted mask of the object and the GT"""
    # Extract the TP/TN/FP/FN regions
    tp = pd_bool_mask * gt_bool_mask
    tn = (1 - pd_bool_mask) * (1 - gt_bool_mask)
    fp = pd_bool_mask * (1 - gt_bool_mask)
    fn = (1 - pd_bool_mask) * gt_bool_mask

    # Compute the IoU
    if tp.sum == 0:
        print(vid_name)
        obx_IoU = tp.sum()/(tp.sum()+fp.sum()+fn.sum())
        print(obx_IoU)
        obx_IoU = 0.0
        print(obx_IoU)
    else:
        obx_IoU = tp.sum()/(tp.sum()+fp.sum()+fn.sum())

    return obx_IoU


# - Arguments ---
parser = ArgumentParser()
parser.add_argument('--model', default='./saves/XMem.pth')

# Data options
parser.add_argument('--d16_path', default='../DAVIS/2016')
parser.add_argument('--d17_path', default='../DAVIS/2017')
parser.add_argument('--y18_path', default='../YouTube2018')
parser.add_argument('--y19_path', default='../YouTube')
parser.add_argument('--lv_path', default='../long_video_set')
parser.add_argument('--burst_path', default='../BURST')
parser.add_argument('--vot_path', default='../VOTS2023')
parser.add_argument('--mose_path', default='../MOSE')
# For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
parser.add_argument('--generic_path')

parser.add_argument('--dataset', help='Available dataset name options are : burst-test / burst-val / d16-val / d17-test / d17-val / generic / lvos-test / lvos-val / mose-val / y18-val / y19-val', default='d17-val')

parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--output', default=None)
parser.add_argument('--save_all', action='store_true',
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

# Long-term memory options
parser.add_argument('--disable_long_term', action='store_true')
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time',
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

# Multi-scale options
parser.add_argument('--save_scores', action='store_true',
                    help="Save the probabilities as well as the logits.")
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=480, type=int,
            help='Resize the shorter side to this size. -1 to use original resolution.')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for the dataloader')
parser.add_argument('--verbose', action='store_false', help='Disabeling icecream prints')
parser.add_argument('--iouatX', type=float, default=0.0, help="Issue a prompt if IoU belove this value")
parser.add_argument('--HatX', type=float, default=1000.0, help="Issue a prompt if entropy belove this value")
parser.add_argument('--derivatX', type=float, default=0.0, help="Issue a prompt is the derivative is above this value")
parser.add_argument('--working_upd', action='store_true', help='Update the WORKING memory with the resulted prompted mask')
parser.add_argument('--deep_upd', action='store_true', help="Update the Deeper memory with the resulted prompted mask")
parser.add_argument('--ff', type=int, default=5, help="Every ff frame update the with the mask")

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']

if args.verbose:
    ic.disable()

args.output = f"{args.output}_Dummy_{args.ff}_working_{args.working_upd}_deep_{args.deep_upd}"

if args.output is None:
    args.output = f'../output/{args.dataset}'
    print(f'Output path not provided. Defaulting to {args.output}')

"""
Data preparation
"""
is_youtube = args.dataset.startswith('Y')
is_davis = args.dataset.startswith('D')
is_lv = args.dataset.startswith('LV')
is_BURST = args.dataset.startswith('B')
is_VOTS2023 = args.dataset.startswith('VOTS')
is_MOSE = args.dataset.startswith('MOSE')


with open(os.path.join('conf','eval_config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

dataset_name = args.dataset
data_cfg = cfg.get('datasets').get(dataset_name)
is_burst = ('burst' in dataset_name)
# setup dataset
image_dir = data_cfg.get('image_directory')
if is_burst:
    json_dir = data_cfg.get('json_directory')
    size_dir = data_cfg.get('size_directory')

if is_burst:
    # BURST style -- masks stored in a json file
    meta_dataset = BURSTTestDataset(image_dir,
                                    json_dir,
                                    size=data_cfg.get('size'),
                                    skip_frames=data_cfg.get('skip_frames'))
    burst_handler = BURSTResultHandler(meta_dataset.json)
else:
    # DAVIS/YouTubeVOS/MOSE style -- masks stored as PNGs
    mask_dir = data_cfg.get('mask_directory')
    first_frame_mask_dir = data_cfg.get('first_mask_directory')
    ic(first_frame_mask_dir)
    subset = data_cfg.get('subset')
    # meta_dataset = VOSTestDataset(image_dir,
    #                               mask_dir,
    #                               use_all_masks=data_cfg.use_all_masks,
    #                               req_frames_json=json_dir,
    #                               size=data_cfg.size,
    #                               size_dir=size_dir,
    #                               subset=subset)
    meta_dataset = VOSTestDataset(image_dir,
                                  mask_dir,
                                  first_frame_mask_dir,
                                #   use_all_masks=data_cfg.get('use_all_masks'),  # default
                                  use_all_masks=True,   # To enable testing with GT masks
                                  req_frames_json=None,
                                  size=data_cfg.get('size'),
                                  size_dir=None,
                                  subset=subset)

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

total_process_time = 0
total_frames = 0

# Start eval
prompt_meta_data = dict()
entropy_seuil = args.HatX  # if above 0.5, use the GT mask
prompt_meta_data["Threshold"] = entropy_seuil
prompt_meta_data["IoU@"] = args.iouatX
prompt_meta_data["Derivative"] = args.derivatX
for vdx, vid_reader in enumerate(progressbar(meta_loader,
                                             max_value=len(meta_dataset),
                                             redirect_stdout=True)):
    # When the process is killed by the OOM killer (Out Of Memory Killer) [https://unix.stackexchange.com/questions/614950/python-programs-suddenly-get-killed]
    if args.dataset == "lvos-val":
        if vdx >= 10:    # when crashing on big datasets
            continue
    # problem :[object 3 appears later, no mask at the start ??] 
    # solution for the 1 st problem: look at the firsr annotation folder of 
    # the sequence. But only initialize with the new object, not all 3...
    # solution to 3 rd problem: look at the first annotation folder
    ic(vid_reader.vid_name)
    # if vid_reader.vid_name in ok: continue
    # TODO for OOM: Better would be to list all available files, and then start from the index -1, to be sure tho have covered everything
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
    first_mask_loaded = False

    prompts_seq_dict = dict()
    for fdx, data in enumerate(loader):
        ic(fdx)
        with (torch.cuda.amp.autocast(enabled=not args.benchmark)):
            rgb = data['rgb'].cuda()[0]
            msk = data.get('mask')
            valid_labels = data.get('valid_labels')
            if valid_labels is not None:
                valid_labels = valid_labels.tolist()

            first_msk = data.get('first_mask')
            first_valid_labels = data.get('first_valid_labels')
            if first_msk is not None:
                first_valid_labels = first_valid_labels.tolist()

            ic('XXXXXXXXXXXXXXXXXXXX')
            ic(rgb)
            ic(msk)
            ic(first_msk)
            ic(first_valid_labels)
            ic(valid_labels)
            ic('XXXXXXXXXXXXXXXXXXXX')

            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['resize_needed']
            path_to_image = info['path_to_image']

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

            if args.flip:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            if (first_msk is not None) and args.flip:
                first_msk = torch.flip(first_msk,
                                       dims=[-1]) if first_msk is not None else None

            ic("YYYYYYYYYYYYYYYYYYYY")
            # # Map possibly non-continuous labels to continuous ones
            # if msk is not None:
            #     # In this region, add the missing frame for the first problem
            #     # sequence
            #     msk, labels = mapper.convert_mask(msk[0].numpy(),
            #                                       exhaustive=True)
            #     ic(msk.shape)
            #     ic(labels)
            #     msk = torch.Tensor(msk).cuda()
            #     ic(msk.shape)
            #     if not is_burst:
            #         if need_resize:
            #             msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
            #     processor.set_all_labels(list(mapper.remappings.values()))
            # else:
            #     labels = None

            # Map possibly non-continuous labels to continuous ones
            if first_msk is not None:
                # In this region, add the missing frame for the first problem 
                # sequence
                msk, labels = mapper.convert_mask(first_msk[0].numpy(),
                                                  exhaustive=True)
                ic(1)
                ic(msk)
                ic(labels)
                msk = torch.Tensor(msk).cuda()
                ic(msk)
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
                    ic(2)
                    ic(msk)
                    ic(labels)
                    msk = torch.Tensor(msk).cuda()
                    ic(msk)
                    if not is_burst:
                        if need_resize:
                            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                else:
                    labels = None
            ic("YYYYYYYYYYYYYYYYYYYY")

            # TODO ############################################################
            # TODO: WORKING AREA ---
            GT_labels = labels
            GT_msk = msk
            if fdx == 0:
                init_labels_list = [l for l in labels]
                frame_0_labels = init_labels_list
                init_msk = msk
                ic(init_msk)
                prompts_frame_dict = dict()
                for obx in init_labels_list:
                    init_prompt_dict = dict()
                    init_prompt_dict[f"id_{obx}"] = {"prompt": [],
                                                     "fdx": [],
                                                     "H": [],
                                                     "IoU": []}
                    prompts_frame_dict.update(init_prompt_dict)

            if (first_msk is not None) and fdx != 0:
                init_labels_list = [l for l in labels]
                init_msk = msk
                ic(init_msk)
                for obx in init_labels_list:
                    # Don't delelte the info from the first elements
                    if obx in frame_0_labels: continue
                    init_prompt_dict = dict()
                    init_prompt_dict[f"id_{obx}"] = {"prompt": [],
                                                     "fdx": [],
                                                     "H": [],
                                                     "IoU": []}
                    prompts_frame_dict.update(init_prompt_dict)
                frame_0_labels = init_labels_list

            if fdx != 0 and first_msk is None:
                msk = None
                labels = None
            
            ic('ZZZZZZZZZZZZZZZZZZZZZZZZ')
                
            # Adding a new object idx in the current space
            # if fdx != 0 and first_frame_mask_dir is not None:
            #     if first_msk is not None:
            #         prob = processor.segment_fdx(rgb, first_msk, labels,
            #                              end=(fdx == vid_length-1))
            # else:
                # Predict a segmentation mask for the current frame
            ic(msk)
            ic(rgb)
            prob = processor.segment_fdx(rgb, msk, labels,
                                         end=(fdx == vid_length-1))
            
            ic(prob)
            ic('AAAAAAAAAAA')
            prob_shape_for_subtiture_mask = prob.shape
            
            prob_u_og = prob.clone().detach().cpu().numpy()
            out_mask_og = torch.max(prob, dim=0).indices
            out_mask_og = (out_mask_og.detach().cpu().numpy()).astype(np.uint8)

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape,
                                     mode='bilinear', align_corners=False)[:,0]
            if args.flip:
                prob = torch.flip(prob, dims=[-1])
                
            ic(prob.shape)

            prob_u = prob.clone().detach().cpu().numpy()
            out_mask = torch.max(prob, dim=0).indices
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            # Depending on the Entropy results: Look at the mask as a new prob_u/out_mask
            stat_api.entropy = prob_u
            stat_api.norm_entropy = prob_u
            efficient_entropy = stat_api.norm_entropy

            # Filter the entropy based on the object, and act on it.
            calls_for_obx = []
            attached_entropy = []  # Use this one for the derivative
            for obx in init_labels_list:
                obj_mask = out_mask == obx
                masked_entropy_odx = get_masked_entropy(obj_mask,
                                                        efficient_entropy, 5)
                masked_entropy_odx = masked_entropy_odx.sum()/obj_mask.sum() if obj_mask.sum() != 0 else 0
                attached_entropy.append(masked_entropy_odx)
                
                # Considere the IoU or other similar metric to issue a prompt
                ic(GT_msk.shape)
                GT_obx_mask = GT_msk[obx-1, :, :].detach().cpu().numpy().astype(bool)
                GT_obx_mask_copy = GT_msk
                if need_resize:
                    GT_obx_mask_copy = F.interpolate(GT_msk.unsqueeze(1), shape,
                                                    mode='bilinear', align_corners=False)[:,0]
                
                GT_obx_mask_copy = GT_obx_mask_copy[obx-1, :, :].detach().cpu().numpy().astype(bool)
                ic(GT_obx_mask.shape)
                ic(pad_divide_by(GT_msk[obx-1, :, :], 16)[0].shape)
                ic(obj_mask.shape)
                ic(pad_divide_by(torch.Tensor(obj_mask), 16)[0].shape)
                ic(out_mask.shape)
                ic(GT_obx_mask_copy.shape)
                obx_IoU = get_IoU(obj_mask, GT_obx_mask_copy)
                ic(obx_IoU)
                ic(args.iouatX)
                
                # Depending on how good the entropy is, act upon it.
                prompt_flag = (obx_IoU < args.iouatX) or (masked_entropy_odx > entropy_seuil)
                # Dummy_prompt
                dummy_condition = 0 == fdx % args.ff
                prompt_flag = dummy_condition
                
                calls_for_obx.append(prompt_flag)
                ic(vid_name, fdx, obx, masked_entropy_odx, prompt_flag)  # Summary statistics

                prompts_frame_dict[f"id_{obx}"] = {"prompt": prompts_frame_dict[f"id_{obx}"]["prompt"] + [prompt_flag],
                                                   "fdx": prompts_frame_dict[f"id_{obx}"]["fdx"] + [fdx],
                                                   "H": prompts_frame_dict[f"id_{obx}"]["H"] + [masked_entropy_odx],
                                                   "IoU": prompts_frame_dict[f"id_{obx}"]["IoU"] + [obx_IoU]}
                prompts_seq_dict.update(prompts_frame_dict)
                
            if fdx == 0:
                # Can only perform an update of the memory is already a mask is predicted... else nothing to do, Hence no interaction for the first frame seen
                calls_for_obx = [False for _ in calls_for_obx]

            # Use the GT mask as a proxy for inputing a "new predicted mask"
            # TODO: Later on, switch to SAM or other pipeline type
            ic(calls_for_obx)
            if max(calls_for_obx):
                substitute_mask = torch.ones([len(init_labels_list), *prob_shape_for_subtiture_mask[1:]])
                ic(substitute_mask.shape, substitute_mask.dtype)
                # Add the mask from prediction to the subsitute_mask
                for e in init_labels_list:
                    one_hot_mask = e == out_mask_og
                    ic(out_mask.shape)
                    ic(out_mask_og.shape)
                    ic(one_hot_mask.shape)
                    substitute_mask[e-1, :, :] = torch.from_numpy(one_hot_mask)

                ic(substitute_mask.shape)
                ic(prob_shape_for_subtiture_mask)

                for obx, need_prompt in enumerate(calls_for_obx):
                    # Replace the mask from the prediction with the mask from the GT
                    if need_prompt:
                        substitute_mask[obx, :, :] = GT_msk[obx, :, :]

                # Update the model's data with the substitute mask
                ic(substitute_mask.shape)
                ic(GT_labels)
                ic(init_labels_list)
                ic(init_msk.shape)
                # GT_labels = [l for l in GT_labels]
                # ic(GT_labels)
                prob_sub = processor.update_with_an_input_mask_w_memory_update(substitute_mask,
                                                                               GT_labels,
                                                                               args.working_upd,
                                                                               args.deep_upd)

                # Upsample to original size if needed
                if need_resize:
                    prob_sub = F.interpolate(prob_sub.unsqueeze(1), shape,
                                             mode='bilinear', align_corners=False)[:,0]
                if args.flip:
                    prob_sub = torch.flip(prob_sub, dims=[-1])

                out_mask_sub = torch.max(prob_sub, dim=0).indices
                out_mask_sub = (out_mask_sub.detach().cpu().numpy()).astype(np.uint8)
                out_mask = out_mask_sub

            # Update the memory
            processor.update_memory_state()
            
            
            # if fdx == 1:
            #     quit()

            # TODO: WORKING AREA ---
            # TODO ############################################################

            # Work with logits
            # pred_logits = processor.get_logits()
            # if pred_logits is not None:
            #     pred_logits = pred_logits[0].clone().detach().cpu().numpy()

            # Save the mask
            if args.save_all or info['save'][0]:
                out_path = path.join(args.output, dataset_name, 'Annotations')
                this_out_path = path.join(out_path, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if args.save_scores:
                np_path_softmax = path.join(args.output, dataset_name,
                                            'softmax', vid_name)
                # np_path_logits = path.join(args.output, dataset_name, 'logits', vid_name)
                os.makedirs(np_path_softmax, exist_ok=True)
                # os.makedirs(np_path_logits, exist_ok=True)
                if fdx == len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path_softmax, f'backward.hkl'), mode='w') # What does that do again ??
                if args.save_all or info['save'][0]:
                    # hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')
                    # ic(prob_u)
                    hkl.dump(prob_u, path.join(np_path_softmax, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')
                    # if pred_logits is not None:
                    #     hkl.dump(pred_logits, path.join(np_path_logits, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')
        
    # Save the prompts results
    np_path_prompt = path.join(args.output, dataset_name, 'prompts', vid_name)
    os.makedirs(np_path_prompt, exist_ok=True)
    for key, value in prompts_seq_dict.items():
        df = pl.DataFrame(value)
        file_name = os.path.join(np_path_prompt, f"{key}.parquet")
        df.write_parquet(file_name)
    # Convert the json into a polar dataframe and save as parquet file for each object

    # prompts_issued_dict.update({vid_name: prompts_seq_dict})

print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

print(prompt_meta_data)
df = pl.DataFrame(prompt_meta_data)
file_name = os.path.join(args.output, dataset_name, 'prompts', "meta_data.parquet") # TODO: use json instead
df.write_parquet(file_name)

# import json
# # with open('prompts_issued.json', 'w') as f:
# with open('test_issued.json', 'w') as f:
#     json.dump(prompts_issued_dict, f)

# if not args.save_scores:
#     if is_youtube:
#         print('Making zip for YouTubeVOS...')
#         shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output, 'Annotations')
#     elif is_davis and args.dataset.split('-')[-1] == 'test':
#         print('Making zip for DAVIS test-dev...')
#         shutil.make_archive(args.output, 'zip', args.output)
