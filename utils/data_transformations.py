import torch


def make_a_flip(args, rgb, msk, first_msk):
    if args.flip:
        rgb = torch.flip(rgb, dims=[-1])
        msk = torch.flip(msk, dims=[-1]) if msk is not None else None
        
        if first_msk is not None:
            first_msk = torch.flip(first_msk, dims=[-1]) if first_msk is not None else None
        
    return rgb, msk, first_msk

def data_info(data):
    rgb = data['rgb'].cuda()[0]
    rgb_raw = data['rgb_raw']
    msk = data.get('mask').clone()
    valid_labels = data.get('valid_labels')
    if valid_labels is not None:
        valid_labels = valid_labels.tolist()

    first_msk = data.get('first_mask')
    first_valid_labels = data.get('first_valid_labels')
    if first_msk is not None:
        first_valid_labels = first_valid_labels.tolist()

    info = data['info']
    frame = info['frame'][0]
    shape = info['shape']
    need_resize = info['resize_needed']
    path_to_image = info['path_to_image']
    return rgb,rgb_raw,msk,first_msk,info,frame,shape,need_resize

def resize_entropy_map_for_SAM(resize_transform_1024, fdx_curve_recorder, obx_pseudo):
    _masked_entropy_map_tensor = torch.from_numpy(fdx_curve_recorder.entropy_memory_frame_map[obx_pseudo+1][0])
    return resize_transform_1024(_masked_entropy_map_tensor[None,None,:,:])[0,0]

def adapt_img_gt_mask_pd_mask_for_SAM(rgb_raw, GT_msk, resize_from_sVOS_baseline_to_SAM, obx, _Xmem_mask_obx_tensor):
    resized_gt_mask_for_irobot = resize_from_sVOS_baseline_to_SAM(GT_msk[obx, :, :][None,None,:,:])[0,0]
    _Xmem_mask_obx_tensor_SAM = resize_from_sVOS_baseline_to_SAM(_Xmem_mask_obx_tensor[None,None,:,:])[0,0]
    rgb_raw_for_SAM = resize_from_sVOS_baseline_to_SAM(rgb_raw.permute(0,3,1,2)).permute(0,2,3,1)
    return _Xmem_mask_obx_tensor_SAM,resized_gt_mask_for_irobot,rgb_raw_for_SAM