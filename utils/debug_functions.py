import os
import matplotlib.pyplot as plt
import seaborn as sns


def debug_show_entropy(SAM_debugger, vid_name, fdx, rgb_raw, obx, normalized_entropy_array_edition, XMem_mask, masked_entropy_obx_map, obx_gt_mask, obx_IoU):
    obj_mask = XMem_mask == obx
    fig_name = f"{vid_name}/entropy_map_norm_masked"
    SAM_debugger.save_entropy(masked_entropy_obx_map[obx][0], fig_name, f"{fdx}_{obx}")
                
                    # Save the entropy mask if debug mode
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.heatmap(masked_entropy_obx_map[obx][0], cmap='magma', cbar=False)
                    
    directory_to_save_entropy = os.path.join("Entropy_debug",vid_name)
    os.makedirs(directory_to_save_entropy, exist_ok=True)
    plt.savefig(f'{directory_to_save_entropy}/masked_entropy_obx_map_{fdx}_{obx}.png')
                    
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.heatmap(normalized_entropy_array_edition[0], cmap='magma', cbar=False)
                    
    directory_to_save_entropy = os.path.join("Entropy_debug",vid_name)
    os.makedirs(directory_to_save_entropy, exist_ok=True)
    plt.savefig(f'{directory_to_save_entropy}/mentropy_obx_map_{fdx}_{obx}.png')
                    
    obx_under = 0.1
    if obx_IoU < obx_under:
        SAM_debugger.set_name_for_figure(f"{vid_name}_{fdx}_{obx}_under_{obx_under}")
        SAM_debugger.show_res(rgb_raw.detach().cpu().numpy()[0], [obj_mask], color=[45,179,195])
        SAM_debugger.set_name_for_figure(f"{vid_name}_{fdx}_{obx}_under_{obx_under}_GT")
        SAM_debugger.show_res(rgb_raw.detach().cpu().numpy()[0], [obx_gt_mask], color=[255,215,0])
        
        
        
def debug_phase_2(mask_refiner, SAM_debugger, vid_name, fdx, GT_msk, resize_from_sVOS_baseline_to_SAM, obx, normalized_entropy_tensor_edition, obx_pseudo, _Xmem_mask_obx_tensor_SAM, rgb_raw_for_refinement):
    SAM_debugger.set_name_for_figure(f"{vid_name}_{fdx}_{obx_pseudo}_A_pseudo")
    print(normalized_entropy_tensor_edition)
    normalized_entropy_tensor_edition = resize_from_sVOS_baseline_to_SAM(normalized_entropy_tensor_edition[None,None,:,:])[0][0]
    S_map_2 = normalized_entropy_tensor_edition.detach().cpu().numpy()
    SAM_debugger.show_res(rgb_raw_for_refinement, [_Xmem_mask_obx_tensor_SAM.cpu().numpy()], color=[45,179,195], S_map=S_map_2)
                        # Display the GT
    SAM_debugger.set_name_for_figure(f"{vid_name}_{fdx}_{obx}_GT")
    gt_mask_for_irobot = GT_msk[obx_pseudo, :, :]
    resized_gt_mask_for_irobot = resize_from_sVOS_baseline_to_SAM(gt_mask_for_irobot[None,None,:,:])[0,0]
    SAM_debugger.show_res(rgb_raw_for_refinement, [resized_gt_mask_for_irobot.cpu().numpy()], color=[255,215,0])
    mask_refiner.save_results_under(f"{vid_name}_{fdx}_{obx_pseudo}_B_pseudo")
    return normalized_entropy_tensor_edition


def debug_phase_3(SAM_debugger, vid_name, fdx, resize_from_sVOS_baseline_to_SAM, obx, normalized_entropy_tensor_edition, _Xmem_mask_obx_tensor_SAM, rgb_raw_for_refinement, resized_gt_mask_for_irobot):
    SAM_debugger.set_name_for_figure(f"{vid_name}_{fdx}_{obx}_A")
    SAM_debugger.show_res(rgb_raw_for_refinement, [_Xmem_mask_obx_tensor_SAM.cpu().numpy()], color=[45,179,195])
    print(normalized_entropy_tensor_edition)
    normalized_entropy_tensor_edition = resize_from_sVOS_baseline_to_SAM(normalized_entropy_tensor_edition[None,None,:,:])[0][0]
    S_map_2 = normalized_entropy_tensor_edition.detach().cpu().numpy()
    SAM_debugger.show_res(rgb_raw_for_refinement, [_Xmem_mask_obx_tensor_SAM.cpu().numpy()], color=[45,179,195], S_map=S_map_2)
                        # Display the GT
    SAM_debugger.set_name_for_figure(f"{vid_name}_{fdx}_{obx}_GT")
    SAM_debugger.show_res(rgb_raw_for_refinement, [resized_gt_mask_for_irobot.cpu().numpy()], color=[255,215,0])