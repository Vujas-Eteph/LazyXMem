"""
This is the base XMem, but with the unsctructered core.

Quick use:
- python3 eval_XMem_base.py --output ../output_for_inference/Single_Models/base_XMem --save_score

Nota Bene: Currently wokrs only with datasets that have gt mask in the dataset (validation/training essentially),
as I'm loading the masks to use an interaction in the first place. Could be replaced/altered in the futur tho.

Modified by Stéphane Vujasinovic
"""

# When the process is killed by the OOM killer (Out Of Memory Killer) [https://unix.stackexchange.com/questions/614950/python-programs-suddenly-get-killed]


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

from argument_loader import BaseArgParser


# - FUNCTION ---
def flip_input(
    gt_rgb: torch.Tensor,
    gt_msk,
    first_msk
):
    gt_rgb = torch.flip(gt_rgb, dims=[-1])
    gt_msk = torch.flip(gt_msk, dims=[-1]) if gt_msk is not None else None
    first_msk = torch.flip(first_msk, dims=[-1]) if first_msk is not None else None

    return gt_rgb, gt_msk, first_msk

def load_msk_data(data, str_1='mask', str_2='valid_labels'):
    msk = data.get(str_1)
    labels = data.get(str_2)
    if labels is not None:
        labels = labels.tolist()
    return msk, labels


# - MAIN ---
if "__main__" == __name__:
    # - Arguments ---
    args = BaseArgParser()    
    args = args.arguments_parser()
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']

    if args.verbose:
        ic.disable()

    if args.output is None:  # TODO: As not anymore the default behavior  Change this
        raise("TRASH as no output arguments is given")
        #args.output = f'../output/{args.dataset}'
        #print(f'Output path not provided. Defaulting to {args.output}')

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
    FLIP_FLAG = args.flip
    SAVE_OUTPUT_FLAG = args.save_all
    SAVE_SCORES_FLAG = args.save_scores
    for vdx, vid_reader in enumerate(progressbar(meta_loader,
                                                max_value=len(meta_dataset),
                                                redirect_stdout=True)):
        
        #print(vdx)
        #if vdx < 80:
        #    continue
        
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
        
        #print(vid_name)

        mapper = MaskMapper()
        processor = DestructuredInferenceCore(network, config=config)
        first_mask_loaded = False
        pmf_list = []
        for fdx, data in enumerate(loader):
            with (torch.cuda.amp.autocast(enabled=not args.benchmark)):
                # - DATA PREPARATION ---
                # Load data
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['resize_needed']
                path_to_image = info['path_to_image']
                gt_rgb = data['rgb'].cuda()[0]
                gt_msk, valid_labels = load_msk_data(data, 'mask', 'valid_labels')
                first_msk, first_valid_labels = load_msk_data(data, 'first_mask', 'first_valid_labels')

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
                    if gt_msk is not None:
                        first_mask_loaded = True
                    else:
                        continue

                if FLIP_FLAG:
                    gt_rgb, gt_msk, first_msk = flip_input(gt_rgb, gt_msk, first_msk)

                # TODO: Check if this deals with the fact that no annotations frame is needed in the val set ?
                # - SEMI-AUTOMATIC VIDEO OBJECT TRACKING/SEGMENTATION ---
                # Map possibly non-continuous labels to continuous ones
                # In this region, add the missing frame for the first problem
                # sequence
                # TODO: Might be able to reduce this even further
                if fdx == 0:
                    # Initialize
                    gt_msk, labels = mapper.convert_mask(gt_msk[0].numpy(),
                                                        exhaustive=True)
                    gt_msk = torch.Tensor(gt_msk).cuda()
                    if not is_burst:
                        if need_resize:
                            gt_msk = vid_reader.resize_mask(gt_msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                elif (first_msk is not None) and fdx != 0:
                    # Add new object to be tracked
                    gt_msk, labels = mapper.convert_mask(first_msk[0].numpy(),
                                                    exhaustive=True)
                    gt_msk = torch.Tensor(gt_msk).cuda()
                    if not is_burst:
                        if need_resize:
                            gt_msk = vid_reader.resize_mask(gt_msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                else:
                    # Normal Tracking (Cover the dataloader's mask and labels)
                    gt_msk = None
                    labels = None

                pmf = processor.segment_fdx(gt_rgb, gt_msk, labels,
                                            end=(fdx == vid_length-1))
                processor.update_memory_state()

                # Upsample to original size if needed
                if need_resize:
                    pmf = F.interpolate(pmf.unsqueeze(1), shape,
                                        mode='bilinear', align_corners=False)[:,0]
                if FLIP_FLAG:
                    pmf = torch.flip(pmf, dims=[-1])
                cloned_pmf = pmf.clone().detach().cpu().numpy()
                pmf_list.append(cloned_pmf)
                pd_mask = torch.max(pmf, dim=0).indices
                pd_mask = (pd_mask.detach().cpu().numpy()).astype(np.uint8)

                # - RECORD RESULTS ---
                # Save masks
                if SAVE_OUTPUT_FLAG or info['save'][0]:
                    out_path = path.join(args.output, dataset_name, 'Annotations')
                    this_out_path = path.join(out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)
                    pd_mask = mapper.remap_index_mask(pd_mask)
                    out_img = Image.fromarray(pd_mask)
                    if vid_reader.get_palette() is not None:
                        out_img.putpalette(vid_reader.get_palette())
                    out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

        # Save probability mass maps
        if SAVE_SCORES_FLAG:
            np_path_softmax = path.join(args.output + "Base", dataset_name,
                                        'softmax', vid_name)
            os.makedirs(np_path_softmax, exist_ok=True)
            Z = np.array(pmf_list)
            np.save(os.path.join(np_path_softmax, "pmf") + ".npy", Z)
            if fdx == len(loader)-1:
                hkl.dump(mapper.remappings, path.join(np_path_softmax, f'backward.hkl'), mode='w') # TODO: What does that do again ??

    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
