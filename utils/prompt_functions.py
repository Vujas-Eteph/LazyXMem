import numpy as np

from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns


def init_prompt_tracker(init_labels_list, prompts_frame_dict):
    for obx in init_labels_list:
        init_prompt_dict = {}
        init_prompt_dict[f"id_{obx}"] = {"prompt": [], "fdx": [], "H": [], "IoU": []}
        prompts_frame_dict.update(init_prompt_dict)


def add_new_obj_to_prompt_tracker(prompts_frame_dict, obx):
    init_prompt_dict = dict()
    init_prompt_dict[f"id_{obx}"] = {"prompt": [], "fdx": [], "H": [], "IoU": []}
    prompts_frame_dict.update(init_prompt_dict)


def add_new_entry_to_prompt_tracker(prompts_seq_dict, fdx, prompts_frame_dict, obx, masked_entropy_obx, obx_IoU, prompt_flag):
    prompts_frame_dict[f"id_{obx}"] = {"prompt": prompts_frame_dict[f"id_{obx}"]["prompt"] + [prompt_flag],
                                                   "fdx": prompts_frame_dict[f"id_{obx}"]["fdx"] + [fdx],
                                                   "H": prompts_frame_dict[f"id_{obx}"]["H"] + [masked_entropy_obx[obx]],
                                                   "IoU": prompts_frame_dict[f"id_{obx}"]["IoU"] + [obx_IoU]}
    prompts_seq_dict.update(prompts_frame_dict)


def call_a_prompt_for_obx(obx, args, entropy_seuil, fdx_curve_recorder, obx_IoU, entropy_deriv_odx):
    prompt_flag = (obx_IoU < args.iouatX) or (fdx_curve_recorder.entropy_memory_frame[obx] > entropy_seuil) or (entropy_deriv_odx > args.derivatX if entropy_deriv_odx is not None else False)
    pseudo_prompt_flag = (fdx_curve_recorder.entropy_memory_frame[obx] > args.pseudo_entropy_seuil) or (args.derivatX > entropy_deriv_odx > args.pseudo_derivatX if entropy_deriv_odx is not None else False)
    if prompt_flag:
        pseudo_prompt_flag = False
                
    fdx_curve_recorder.call_pseudo_inter_for_obx.append(pseudo_prompt_flag)
    fdx_curve_recorder.call_user_inter_for_obx.append(prompt_flag)


def adapt_prompts_for_the_first_frame(calls_for_obx_user, calls_for_obx_pseudo):
    # Can only perform an update of the memory is already a mask is predicted... else nothing to do, Hence no interaction for the first frame seen
    calls_for_obx_user = [False for _ in calls_for_obx_user]
    calls_for_obx_pseudo = [False for _ in calls_for_obx_pseudo]


def record_entropy_values(entropy_memory, fdx, entropy_memory_frame):
    entropy_memory[fdx] = entropy_memory_frame
    
    
def get_prompt_for_sam(args, robot_click_coords):
    input_coords = None
    input_labels = None
    BBox = None
    if args.user_pseudo_BBox:
        BBox = np.array([robot_click_coords['BBox']])
    else:    
        input_coords = np.array([[robot_click_coords['w'], robot_click_coords['h']]])
        interaction_value = 1
        input_labels = np.array([interaction_value])
    return input_coords,input_labels, BBox


def find_user_interaction_coords(zivos_robot, _Xmem_mask_obx_tensor_SAM, resized_gt_mask_for_irobot):
    robot_click_coords = zivos_robot.get_interaction(resized_gt_mask_for_irobot,
                                                 resized_gt_mask_for_irobot,
                                                 mode="TP")
    if robot_click_coords['w'] is None:  # Because there is no object add a negative interaction instead
        robot_click_coords = zivos_robot.get_interaction(resized_gt_mask_for_irobot,
                                                     _Xmem_mask_obx_tensor_SAM,
                                                     mode="FP")

    return robot_click_coords


def find_pseudo_interaction_coords(fdx, args, zivos_robot, init_labels_list,
                                   _Xmem_mask_obx_tensor_SAM,
                                   _masked_entropy_map_tensor_SAM,
                                   coeff= 1, debug=False):
    confidence_mask = (1 - _masked_entropy_map_tensor_SAM)

    # Here we leverage the rivos roboter to generate our pseudo interactions
    if args.mode == 0:
        robot_click_coords = zivos_robot.get_interaction(_Xmem_mask_obx_tensor_SAM, _Xmem_mask_obx_tensor_SAM, mode="TP")
    elif args.mode == 1:  # best results
        robot_click_coords = zivos_robot.get_interaction_weighted(_Xmem_mask_obx_tensor_SAM, _Xmem_mask_obx_tensor_SAM, coeff*confidence_mask, mode="TP")  # Higher coeff for weighting more strongly the confidence
    elif args.mode == 2:
        robot_click_coords = zivos_robot.get_interaction(confidence_mask > 1/len(init_labels_list), _Xmem_mask_obx_tensor_SAM, mode="TP")
    else:
        assert("This mode is not supported")
        
    if debug:
        edf_map = compute_euclidian_distance(_Xmem_mask_obx_tensor_SAM)
        final = confidence_mask.cpu().numpy()*edf_map
        save_as_intermediate_figures(fdx, confidence_mask.cpu().numpy(), edf_map, final) # for debugging

    return robot_click_coords


def save_as_intermediate_figures(fdx, confidence_map, edf_map, final):
    print(f'Does this go here???? {fdx}')
    
    plot_heatmap(fdx, confidence_map, name="conf", color_map="viridis")
    plot_heatmap(fdx, edf_map, name="edf", color_map="cividis")
    plot_heatmap(fdx, final, name="final", color_map="mako")
    
    print("Yes it did")

def plot_heatmap(fdx, array, name="", color_map="viridis"):
    plt.figure(figsize=(8, 8))  # Set figure size
    sns.heatmap(array, annot=False, cmap=color_map, cbar=False)  # Heatmap with colorbar
    # plt.title(f"Heatmap: {name}")
    
    # Save figure to the "figures" directory
    plt.savefig(f"figures/{name}_{fdx}.png")
    plt.close()  # Close the figure to avoid displaying it when running in loops


def compute_euclidian_distance(msk: np.ndarray) -> np.ndarray:
    
    msk_p = np.pad(msk, (1, 1), mode="constant", constant_values=(False, False))
    edf_mask_p = ndimage.distance_transform_edt(msk_p)
    edf_mask = edf_mask_p[1:-1, 1:-1]  # unpadding
    
    return edf_mask