"""
Adapt SAM's and sVOS backbone's output/input for better communications between each element

by Stéphane Vujasinovic
"""

# - IMPORTS ---
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


# - FUNCTIONS ---
def prep_transformation_and_image_for_SAM_and_XMem(rgb, rgb_raw):
    resize_transform_OG = transforms.Resize(rgb_raw.shape[1:3],interpolation=InterpolationMode.BICUBIC)
    resize_from_sVOS_baseline_to_SAM = transforms.Resize((1024, 1024),interpolation=InterpolationMode.BICUBIC)
    resize_from_sVOS_baseline_to_SAM_bilinear_variant = transforms.Resize((1024, 1024),interpolation=InterpolationMode.BILINEAR)
    resize_from_SAM_to_sVOS_baseline = transforms.Resize(rgb.shape[1:3],interpolation=InterpolationMode.BICUBIC)
                            
    return resize_from_sVOS_baseline_to_SAM, resize_from_sVOS_baseline_to_SAM_bilinear_variant, resize_from_SAM_to_sVOS_baseline