'''
Entropy Module
Helps generate the entropy results and stuff like that

Stéphane Vujasinovic
'''

# - IMPORTS ---
import numpy as np
from typing import Callable, Tuple
import cv2
import torch
import torch.nn.functional as F


# - CLASS ---
class EntropyHelper:
    """
    Handle entropy calculation.
    """
    def __init__(self):
        pass

    @staticmethod
    def check_if_pmf(
        input: np.ndarray,
        axis: int
    ) -> bool:
        return (1, 1) == (np.round(np.sum(input, axis=axis).min(), 5),
                          np.round(np.sum(input, axis=axis).max(), 5))

    @staticmethod
    def self_information(
        x: np.ndarray
    ) -> np.ndarray:
        return -1*np.log(x)

    @staticmethod
    def compute_entropy(
        x: np.ndarray
    ) -> np.ndarray:
        z = np.zeros([1, *x.shape[1:]], dtype=x.dtype)
        for y in x:
            z += y * EntropyHelper.self_information(y)
        return z

    @staticmethod
    def squeeze_list_of_arrays(
        input_list: list
    ) -> list:
        return [np.squeeze(element) for element in input_list]


    @staticmethod
    def compute_normalized_entropy(
        prediction: np.ndarray
    ):
        assert isinstance(prediction, np.ndarray)
        assert EntropyHelper.check_if_pmf(prediction, 0)
        
        entropy = EntropyHelper.compute_entropy(prediction)
        norm_entropy = entropy / np.log(prediction.shape[0])
        
        array_edition = np.round(norm_entropy, 4)
        tensor_edition = torch.tensor(array_edition[0])
        
        return array_edition, tensor_edition 


    @staticmethod
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

    @staticmethod
    def get_masked_entropy_for_obx(
        pd_mask: np.ndarray,
        obx: int,
        entropy_map: np.ndarray,
        kernel_size: int,
        debug=False
    ) -> np.ndarray:
        obx_mask = pd_mask == obx
        # Dilate the mask based on the kernel
        kernel = EntropyHelper.gen_kernel(kernel_size, obx_mask.sum())
        dilated_mask = cv2.dilate(obx_mask.astype(np.uint8),
                                  kernel,
                                  iterations=1).astype(bool)
        masked_entropy_for_fdx_and_obx_map = entropy_map * dilated_mask

        if debug:
            cv2.imshow("dilates_mask", dilated_mask.astype(np.uint8) * 255)
            cv2.imshow("not_dilates_mask", obx_mask.astype(np.uint8) * 255)

            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                
        masked_entropy_for_fdx_and_obx_mean = EntropyHelper.compute_average_masked_entropy_of_obx(masked_entropy_for_fdx_and_obx_map, obx_mask)

        return masked_entropy_for_fdx_and_obx_map, masked_entropy_for_fdx_and_obx_mean
    
    @staticmethod
    def compute_average_masked_entropy_of_obx(masked_entropy_for_fdx_and_obx_map, obx_mask): 
        return masked_entropy_for_fdx_and_obx_map.sum()/obx_mask.sum() if obx_mask.sum() != 0 else 0


def operation_on_Entropy_over_TPTNFPFN(
    h: np.ndarray,
    TP: np.ndarray,
    TN: np.ndarray,
    FP: np.ndarray,
    FN: np.ndarray,
    op_func: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    H.shape: [#frames, 1, H, W]. Hence the second axis can be discarded
    """
    # Trick to avoid meaning with 0. Take the lowest point of the H, as when no filterning nothing happens, but if a filter was used beforehand is now applied...
    # ... Allows to not take into account the 0 during the mean operaition (no influence on the summation).
    # The trick is to filter the values based on the lowet valuea nd give the filtered version to the op_func
    h_filtered = h[~np.isclose(h, 0.0, rtol=1e-09, atol=1e-09)]
    h_min = (np.array([0]) if h_filtered.tolist() == [] else h_filtered).min()

    TP_H = op_func(h[h*TP >= h_min])
    TN_H = op_func(h[h*TN >= h_min])
    FP_H = op_func(h[h*FP >= h_min])
    FN_H = op_func(h[h*FN >= h_min])

    return TP_H, TN_H, FP_H, FN_H


def sum_Entropy_over_TPTNFPFN(
    H: np.ndarray,
    fdx: int,
    TP: np.ndarray,
    TN: np.ndarray,
    FP: np.ndarray,
    FN: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    H.shape: [#frames, 1, H, W]. Hece the second axis can be discarded
    """
    TP_H = (H[fdx, 0, :, :]*TP).sum()
    TN_H = (H[fdx, 0, :, :]*TN).sum()
    FP_H = (H[fdx, 0, :, :]*FP).sum()
    FN_H = (H[fdx, 0, :, :]*FN).sum()

    return TP_H, TN_H, FP_H, FN_H

def get_IoU(
    pd_bool_mask: np.ndarray,
    gt_bool_mask: np.ndarray
):
    tp = pd_bool_mask * gt_bool_mask
    tn = (1 - pd_bool_mask) * (1 - gt_bool_mask)
    fp = pd_bool_mask * (1 - gt_bool_mask)
    fn = (1 - pd_bool_mask) * gt_bool_mask

    numerator = tp.sum()
    denominator = tp.sum() + fp.sum() + fn.sum()    
    
    if (0 == numerator) or (0 == denominator):
        return 0.0
    
    return numerator/denominator



def get_iou_between_pd_and_gt_obx_mask(shape, need_resize, GT_msk, obx, XMem_mask):
    obx_pd_mask = XMem_mask == obx
    obx_gt_mask = GT_msk
    if need_resize:
        obx_gt_mask = F.interpolate(obx_gt_mask.unsqueeze(1), shape,
                                                     mode='bilinear', align_corners=False)[:,0]

    obx_gt_mask = obx_gt_mask[obx-1, :, :].detach().cpu().numpy().astype(bool)
    
    return obx_gt_mask, get_IoU(obx_pd_mask, obx_gt_mask)



def store_entropy_value_and_compute_derivative_per_object(entropy_memory, fdx, obx, masked_entropy_obx):
    """
    Depending on how good the entropy is, act upon it.
    """
    # TODO: check what happens when object is missing, for instance india, and when a new object is added.
    if bool(entropy_memory):  # Check if dict empty
        # Check if obx is also present in the previous frame
        available_objects_in_prev_frame = entropy_memory[fdx-1].keys()
        if obx in available_objects_in_prev_frame:
            masked_entropy_prev = entropy_memory[fdx-1][obx]
            entropy_deriv_odx = masked_entropy_obx[obx] - masked_entropy_prev
        else:
            entropy_deriv_odx = None
    else:
        entropy_deriv_odx = None
    return entropy_deriv_odx