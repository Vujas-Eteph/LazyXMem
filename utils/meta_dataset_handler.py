"""
TODO: GIVE DESCRIPTION

by Stéphane Vujasinovic
"""


# - IMPORTS ---
from inference.data.burst_test_dataset import BURSTTestDataset
from inference.data.burst_utils import BURSTResultHandler
from inference.data.vos_test_dataset import VOSTestDataset


# - FUNCTIONS ---
def create_meta_dataset(data_cfg, is_burst, image_dir):
    """
    NOTE: THE ONLY FORMAT SUPPORTED (and tested) are DAVIS and LVOS
    """
    if is_burst:
    # BURST style -- masks stored in a json file
        json_dir = data_cfg.get('json_directory')
        size_dir = data_cfg.get('size_directory')
        meta_dataset = BURSTTestDataset(image_dir,
                                    json_dir,
                                    size=data_cfg.get('size'),
                                    skip_frames=data_cfg.get('skip_frames'))
        burst_handler = BURSTResultHandler(meta_dataset.json)
    else:
    # DAVIS/YouTubeVOS/MOSE style -- masks stored as PNGs
        mask_dir = data_cfg.get('mask_directory')
        first_frame_mask_dir = data_cfg.get('first_mask_directory')
        subset = data_cfg.get('subset')
        meta_dataset = VOSTestDataset(image_dir,
                                  mask_dir,
                                  first_frame_mask_dir,
                                #   use_all_masks=data_cfg.get('use_all_masks'),  # default
                                  use_all_masks=True,   # To enable testing with GT masks
                                  req_frames_json=None,
                                  size=data_cfg.get('size'),
                                  size_dir=None,
                                  subset=subset)
                                      
    return meta_dataset