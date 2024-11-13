"""
DO DESCRIPTION

by Stéphane Vujasinovic
"""

# - IMPORTS ---
import numpy as np
from scipy import ndimage

# - FUNCTIONS ---
def exctract_edf(mask):
    msk = np.pad(mask.copy(), (1,1), mode='constant', constant_values=(0, 0))
    edf_msk = ndimage.distance_transform_edt(msk)
    edf_msk = edf_msk[1:-1, 1:-1]
    return edf_msk


def generate_negative_coords(obx, weighted_pmf_per_obx_conf_positive, number_of_objects_identified_in_frame):
    negative_clicks = []
    negative_labels = []
                        
    if number_of_objects_identified_in_frame != 1:
        for obj_neg in np.unique(weighted_pmf_per_obx_conf_positive):
            if obj_neg == obx or obj_neg == 0:
                continue
            edf_msk_neg = exctract_edf(weighted_pmf_per_obx_conf_positive[0] == obj_neg)
            _h, _w = np.where(edf_msk_neg == edf_msk_neg.max())
                                
            negative_clicks.append(np.array([_w[0], _h[0]]))
            negative_labels.append(0)
                            
        # 
        negative_clicks = np.array(negative_clicks)
        negative_labels = np.array(negative_labels)

    return negative_clicks, negative_labels
