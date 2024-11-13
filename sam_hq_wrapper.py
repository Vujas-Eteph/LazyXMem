import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# All my packages
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import segmentation_refinement as refine

from torchvision.transforms.functional import resize  # type: ignore


def refine_prediction_with_SAM(mask_refiner, resize_from_SAM_to_sVOS_baseline, substitute_mask, obx, input_coords, input_labels, BBox=None):
    SAM_pred_mask, score, logits = mask_refiner.refine_mask(point_coords=input_coords, point_labels=input_labels,
                                                            box = BBox, multimask_output= not BBox,
                                                            hq_token_only=True, # use_psp=True, fast=False,
                                                            color=[255,87,51])
                    

    new_mask = resize_from_SAM_to_sVOS_baseline(torch.from_numpy(SAM_pred_mask)[None,:,:,:]).type(substitute_mask.dtype)
    substitute_mask[obx, :, :] = new_mask[0]

class DebugSAM():
    def __init__(self, use_save_feature: bool = False):
        use_save_feature = True
        if use_save_feature:
            self.use_save_feature = use_save_feature
            #self.ROOT_FOLDER = "./DEBUG_VISUALIZATION"
            self.ROOT_FOLDER = "./Visu_paper_arXiv_2"
            self.fig_name = "No_name"
            self.ext = ".jpg"
        pass

    def set_name_for_figure(self, fig_name: str = "No_name"):
        self.fig_name = fig_name

    def set_format_for_figure(self, ext: str = ".jpg"):
        self.ext = ext
        
    @staticmethod
    def show_mask(mask, ax, random_color=False, color=[30,144,255]):
        # color = [250,150,80] # Orange
        # color = [45,179,195] # Blue-Turquoise
        if random_color:
            color_transprante = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color_transprante = np.array([color[0]/255, color[1]/255, color[2]/255, 0.6])
            color_hard = np.array([color[0]/255, color[1]/255, color[2]/255, 1.0])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color_transprante.reshape(1, 1, -1)
        ax.imshow(mask_image)
        # Draw black contour around the mask
        ax.contour(mask, levels=[0.5], colors=[color_hard], linewidths=2)  

    @staticmethod
    def show_points(coords, labels, ax, marker_size=350):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='yellow', marker='*', s=marker_size, edgecolor='black', linewidth=1.5)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='black', linewidth=1.5)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    def show_res(self, image, masks, scores=[None],
                 input_point=None, input_label=None,
                 input_box=None, color=[30,144,255], S_map=None):
        mask = masks[0]
        score = scores[0]
        plt.figure(figsize=(10, 10))
        print(image.shape)
        print(mask.shape)
        if image is not None:
            plt.imshow(image)
        self.show_mask(mask, plt.gca(), color=color)
        if input_box is not None:
            box = input_box[0]
            self.show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            self.show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        if self.use_save_feature:
            path_to_save_fig = os.path.join(self.ROOT_FOLDER,
                                            self.fig_name + self.ext)
            plt.savefig(path_to_save_fig, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

        plt.clf()
        if S_map is not None:
            plt.imshow(S_map, cmap='magma')
            plt.axis('off')
            path_to_save_fig = os.path.join(self.ROOT_FOLDER,
                                            self.fig_name + "S_map" + self.ext)
            plt.savefig(path_to_save_fig, bbox_inches='tight', pad_inches=0)
            plt.clf()
            
    def save_entropy(self, S_map, path, fig_name):
        plt.imshow(S_map, cmap='magma')
        plt.axis('off')
        if not os.path.exists(os.path.join(self.ROOT_FOLDER, path)):
            os.makedirs(os.path.join(self.ROOT_FOLDER, path))
        path_to_save_fig = os.path.join(self.ROOT_FOLDER, path, fig_name + self.ext)
        plt.savefig(path_to_save_fig, bbox_inches='tight', pad_inches=0)
        plt.clf()

    def show_res_multi(self, masks, scores, input_point, input_label, input_box, image):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask, plt.gca(), random_color=True)
        for box in input_box:
            self.show_box(box, plt.gca())
        for score in scores:
            print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.show()


class Mask_Refiner():
    def __init__(self, model_type:str="vit_l", debug_mode:bool=False, use_save_feature:bool=False):
        # SAM-HQ -> https://github.com/SysCV/sam-hq
        sam_checkpoint = os.path.join(os.getcwd(), f"sam-hq/pretrained_checkpoint/sam_hq_{model_type}.pth")
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        
        # CPSP Net -> https://github.com/hkchengrex/CascadePSP
        self.psp_refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'
        
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.debug_sam = DebugSAM(use_save_feature)
            
    def save_results_under(self, fig_name):
        if self.debug_mode:
            self.debug_sam.set_name_for_figure(fig_name)
        

            
    def _refine_with_PSP(self, img: np.ndarray, mask: np.ndarray, fast=False, L=900):
        mask = cv2.cvtColor((mask[0]*255).astype(np.uint8), cv2.IMREAD_GRAYSCALE)[:, :, 0]  # needed
        refined_mask = self.psp_refiner.refine(img, mask, fast=fast, L=L)
        refined_mask = refined_mask.astype(bool)

        return refined_mask[None, :, :]
    
    def _compute_logits_from_mask(self, mask, eps=1e-3):

        def inv_sigmoid(x):
            return np.log(x / (1 - x))

        if self.already_logits:
            logits = mask
        else:
            logits = np.zeros(mask.shape, dtype="float32")
            logits[mask == 1] = 1 - eps
            logits[mask == 0] = eps
            logits = inv_sigmoid(logits)

        # resize to the expected mask shape of SAM (256x256)
        assert logits.ndim == 2
        expected_shape = (256, 256)

        if logits.shape == expected_shape:  # shape matches, do nothing
            pass

        elif logits.shape[0] == logits.shape[1]:  # shape is square
            trafo = ResizeLongestSide(expected_shape[0])
            logits = trafo.apply_image(logits[..., None])

        else:  # shape is not square
            # resize the longest side to expected shape
            trafo = ResizeLongestSide(expected_shape[0])
            logits = trafo.apply_image(logits[..., None])

            # pad the other side
            h, w = logits.shape
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w
            # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
            pad_width = ((0, padh), (0, padw))
            logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

        logits = logits[None]
        assert logits.shape == (1, 256, 256), f"{logits.shape}"
        return logits
    
    def _process_box(self, box, shape, original_size=None, box_extension=0):
        if box_extension == 0:  # no extension
            extension_y, extension_x = 0, 0
        elif box_extension >= 1:  # extension by a fixed factor
            extension_y, extension_x = box_extension, box_extension
        else:  # extension by fraction of the box len
            len_y, len_x = box[2] - box[0], box[3] - box[1]
            extension_y, extension_x = box_extension * len_y, box_extension * len_x

        box = np.array([
            max(box[1] - extension_x, 0), max(box[0] - extension_y, 0),
            min(box[3] + extension_x, shape[1]), min(box[2] + extension_y, shape[0]),
        ])

        if original_size is not None:
            trafo = ResizeLongestSide(max(original_size))
            box = trafo.apply_boxes(box[None], (256, 256)).squeeze()
        return box

    def _compute_box_from_mask(self, mask, original_size=None, box_extension=0):
        coords = np.where(mask == 1)
        min_y, min_x = coords[0].min(), coords[1].min()
        max_y, max_x = coords[0].max(), coords[1].max()
        box = np.array([min_y, min_x, max_y + 1, max_x + 1])

        return self._process_box(box, mask.shape, original_size=original_size, box_extension=box_extension)

    def set_image_for_SAM(self, img: np.ndarray):
        # img: RGB and preferably with the longest side being 1024 for better results (find the refs:)
        
        # Load the RGB image in the class
        self.rgb = img.copy()
        self.predictor.set_image(self.rgb)
    
    def set_mask_for_SAM(self, msk=None):
        """msk: np.ndarray"""      
        self.already_logits = False
        self.msk = msk        
        
    def load_new_mask_as_logits(self, logits):
        self.already_logits = True
        self.msk = logits
    
    def load_new_mask(self, msk):
        self.already_logits = False
        self.msl = msk
        
        
    def refine_mask_using_a_base_SAM_mask(self, sam_mask, point_coords=None, point_labels=None, multimask_output=True, hq_token_only=True):
        masks, scores, logits = self.predictor.predict(point_coords=point_coords,
                                                       point_labels=point_labels,
                                                       mask_input=sam_mask,
                                                       multimask_output=multimask_output, hq_token_only=hq_token_only)

        return masks, scores, logits
    
    def _refine_with_SAM(
        self, 
        point_coords=None, point_labels=None, 
        box=None, 
        mask_input=None, 
        multimask_output=False, hq_token_only=False
    ):
        # Refine the mask with SAM
        mask, scores, logits = self.predictor.predict(point_coords=point_coords,
                                                      point_labels=point_labels,
                                                      box=box,
                                                      mask_input=None if mask_input is None else self._compute_logits_from_mask(mask_input[0]),
                                                      multimask_output=multimask_output,
                                                      hq_token_only=hq_token_only)
        
        if multimask_output:  # take the best mask
            best_mask_id = np.argmax(scores)
            mask = mask[best_mask_id][None]
            scores = scores[best_mask_id][None]
            logits = logits[best_mask_id][None]
            
        return mask, scores, logits
        
    def refine_mask(self,
                    use_sam=True,
                    point_coords=None, point_labels=None, 
                    box=None,
                    multimask_output=False, hq_token_only=False,
                    use_psp=False, fast=True,
                    color = [163,9,210]):
        # IMPORTANT: set hq_token_only to True if single object -> https://github.com/SysCV/sam-hq/blob/9245c85e16d93de200f14734cfa4e9676b2eaad5/demo/demo_hqsam.py#L69
        
        msk = self.msk
        scores = None
        logits = None
        if use_sam:
            msk, scores, logits = self._refine_with_SAM(point_coords, point_labels,
                                                        box, msk,
                                                        multimask_output, hq_token_only)
        if use_psp and msk is not None:
            msk = self._refine_with_PSP(self.rgb, msk, fast)
        
        if self.debug_mode:
            self.debug_sam.show_res(self.rgb, msk, scores, point_coords, point_labels, box, color)
            
        self.already_logits = False
            
        return msk, scores, logits  # refined mask
