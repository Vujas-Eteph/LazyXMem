"""
Inference core decomposed in individual method calls for better manipulation.

by Stéphane Vujasinovic
"""

# - IMPORTS ---
from inference.memory_manager import MemoryManager
from model.network import XMem
from model.aggregate import aggregate
import torch

from util.tensor_util import pad_divide_by, unpad

from typing import Optional, List

from icecream import ic

# - CLASS ---
class DestructuredInferenceCore:
    """
    Destructured Inference Core
    """
    def __init__(
        self,
        network: XMem,
        config
    ):
        self.config = config
        self.network = network
        self.mem_every_fdx = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None
        self.prob_pmf_per_object = None
        self.logits = None

    def clear_memory(
        self
    ):
        self.curr_fdx = -1
        self.last_mem_fdx = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def update_config(
        self,
        config
    ):
        self.mem_every_fdx = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(
        self,
        all_labels
    ):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def _update_internal_fdx(self):
        """
        Increment the internal index of the frame.
        """
        self.curr_fdx += 1
    
    def _update_internal_fdx_bis(self):
        """
        Decrement the internal index of the frame.
        """
        self.curr_fdx -= 1

    def _adjust_internal_flags(
        self,
        mask: Optional[torch.Tensor] = None,
        valid_labels: Optional[List] = None,
        end=False
    ):
        """
        Adjust internal flags to know if a segmentation is needed, a memory
        updates or other stuff.
        """
        is_not_end = not end
        # Decide whether the current frame needs to be segmented
        self.need_segment_flag = \
            (self.curr_fdx > 0) and (
                (valid_labels is None) or
                (len(self.all_labels) != len(valid_labels))
                )

        # Decide whether the current frame should be saved in the memory, and what kind of update to do later (deep, normal)
        self.is_mem_frame_flag = \
            ((self.curr_fdx - self.last_mem_fdx >= self.mem_every_fdx) or
             (mask is not None)) and is_not_end

        # Synchro flags
        is_synchronized_flag = self.deep_update_sync and self.is_mem_frame_flag
        is_no_sync_flag = not self.deep_update_sync and self.curr_fdx - self.last_deep_update_ti >= self.deep_update_every

        # Memory update flags
        self.is_deep_update_flag = (is_synchronized_flag or
                                    is_no_sync_flag) and is_not_end
        self.is_normal_update_flag = (not self.deep_update_sync or
                                      not self.is_deep_update_flag) and is_not_end

    def _extract_features(self, image):
        """
        Extract features
        """
        key, shrinkage, selection, f16, f8, f4 = \
            self.network.encode_key(image, need_sk=self.is_mem_frame_flag,
                                    need_ek=(self.enable_long_term or
                                             self.need_segment_flag)
                                    )
        multi_scale_features = (f16, f8, f4)

        return key, shrinkage, selection, f16, multi_scale_features

    def _segment(
        self,
        _key: torch.Tensor,
        _selection,
        _multi_scale_features: List[torch.Tensor]
    ):
        """Segment a the frame

        Args:
            _key (_type_): _description_
            _selection (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Read memory information
        memory_readout = self.memory.match_memory(_key, _selection)
        memory_readout = memory_readout.unsqueeze(0)

        # Make a prediction for the current frame and propose a new hidden
        # state of the sensory memory
        # Push the encodings through the decoder
        (hidden, logits, pred_prob_with_bg) = \
            self.network.segment(_multi_scale_features,
                                 memory_readout,
                                 self.memory.get_hidden(),
                                 h_out=self.is_normal_update_flag,
                                 strip_bg=False)
        # Test
        self.prob_pmf_per_object = self.network.pmf_per_obj if not None else None
        
        # Logits
        self.logits = logits.clone()

        return hidden, pred_prob_with_bg

    def _adapt_input_mask(
        self,
        _mask: torch.Tensor,
        _valid_labels: Optional[List] = None
    ) -> torch.Tensor:
        """Adapt the mask (either GT or prompted from another method) to
        be compatible with pred_prob_no_bg
        TODO: Actually not sure about that DOCSTRING
        """
        # if we have a predicted mask, we work on it
        # make pred_prob_no_bg consistent with the input mask
        mask_regions = (_mask.sum(0) > 0.5)
        self.pred_prob_no_bg[:, mask_regions] = 0
        # shift by 1 because mask/pred_prob_no_bg do not contain background
        _mask = _mask.type_as(self.pred_prob_no_bg)
        if _valid_labels is not None:
            shift_by_one_non_labels = [i for i in range(self.pred_prob_no_bg.shape[0]) if (i+1) not in _valid_labels]
            # non-labelled objects are copied from the predicted mask
            _mask[shift_by_one_non_labels] = self.pred_prob_no_bg[shift_by_one_non_labels]

        return _mask

    def update_with_an_input_mask(
        self,
        mask: torch.Tensor,
        valid_labels: List
    ):
        """"""
        mask, _ = pad_divide_by(mask, 16)
        mask = self._adapt_input_mask(mask, valid_labels)
        self.pred_prob_with_bg = aggregate(mask, dim=0)
        # self.memory.create_hidden_state(len(self.all_labels), self.key)
        return unpad(self.pred_prob_with_bg, self.pad)

    def update_with_an_input_mask_w_memory_update(
        self,
        mask: torch.Tensor,
        valid_labels: List,
        is_mem_frame_flag=False,
        deep_update_flag=False
    ):
        """"""
        self._update_memory_flags(is_mem_frame_flag, deep_update_flag)
        _, self.shrinkage, _ = self.network.key_proj(self.f16, need_s=self.is_mem_frame_flag, need_e=(self.enable_long_term or self.need_segment_flag))
        mask, _ = pad_divide_by(mask, 16)
        mask = self._adapt_input_mask(mask, valid_labels)
        self.pred_prob_with_bg = aggregate(mask, dim=0)
        return unpad(self.pred_prob_with_bg, self.pad)

    def _update_memory_flags(
        self,
        is_mem_frame_flag=False,
        deep_update_flag=False
    ):
        """
        Force the update of the memory flags
        """
        if is_mem_frame_flag:
            self.is_mem_frame_flag = True
        if deep_update_flag:
            self.is_deep_update_flag = True

    def segment_fdx(
        self,
        image=torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        valid_labels: Optional[List] = None,
        end=False
    ) -> torch.Tensor:
        """"""
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self._update_internal_fdx()
        image, self.pad = pad_divide_by(image, 16)
        self.image = image.unsqueeze(0)  # add the batch dimension

        self._adjust_internal_flags(mask, valid_labels, end)
        (self.key, self.shrinkage, self.selection,
         self.f16, multi_scale_features) = self._extract_features(self.image)

        # segment the current frame if needed
        self.pred_prob_no_bg = None
        self.pred_prob_with_bg = None
        if self.need_segment_flag:
            self.hidden, self.pred_prob_with_bg = \
                self._segment(self.key, self.selection, multi_scale_features)
            # NB: self.hidden is None the first time - should save the previous hidden state   
            # remove batch dim
            self.pred_prob_with_bg = self.pred_prob_with_bg[0]
            self.pred_prob_no_bg = self.pred_prob_with_bg[1:]

        # use the input mask if any
        self.OG_mask = None if mask is None else mask.detach().clone()
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)
            if self.pred_prob_no_bg is not None:
                mask = self._adapt_input_mask(mask, valid_labels)
            self.pred_prob_with_bg = aggregate(mask, dim=0)
            self.pred_prob_no_bg = self.pred_prob_with_bg[1:]

        if self.prob_pmf_per_object is not None:
            # ic.enable()
            # ic(self.prob_pmf_per_object.shape)
            # ic.disable()
            if len(self.prob_pmf_per_object.shape) == 4:
                self.prob_pmf_per_object = self.prob_pmf_per_object.permute(1,0,2,3).squeeze(1)
            elif len(self.prob_pmf_per_object.shape) == 3:
                pass
            else: 
                raise("Error")
            self.prob_pmf_per_object = unpad(self.prob_pmf_per_object, self.pad)

        return unpad(self.pred_prob_with_bg, self.pad)

    def get_logits(self):
        """Extract the logits from the model
        """
        return None if self.logits is None else unpad(self.logits, self.pad)

    def update_memory_state(self):
        """
        When called update the memory of XMem - Sensory, Working and Long Term
        """
        # Update the hidden state of the sensory memory
        if self.need_segment_flag and self.is_normal_update_flag:
            self.memory.set_hidden(self.hidden)
        # also create new hidden states
        if self.OG_mask is not None:
            # Hidden state is of the form [B, n, C_h, H/16, W/16]
            # n is the number of objects
            self.memory.create_hidden_state(len(self.all_labels), self.key)

        # save as memory if needed
        if self.is_mem_frame_flag:
            _hidden_state = self.memory.get_hidden()  # [B, n, C_h, H/16, W/16]
            _prob_mask = self.pred_prob_with_bg[1:].unsqueeze(0)  # [B, n, H, W]
            value, self.hidden = \
                self.network.encode_value(self.image, self.f16,
                                          _hidden_state, _prob_mask,
                                          is_deep_update=self.is_deep_update_flag)
            # value [B, n, C_v, H/16, W/16]
            # self.hidden [B, n, C_h, H/16, W/16]
            # selection [1, C_k, H/16, W/16]
            self.memory.add_memory(self.key,
                                   self.shrinkage,
                                   value,
                                   self.all_labels,
                                   selection=self.selection if self.enable_long_term else None)
            self.last_mem_fdx = self.curr_fdx

            if self.is_deep_update_flag:
                self.memory.set_hidden(self.hidden)
                self.last_deep_update_ti = self.curr_fdx

    def only_update_hidden_state(self):
        """
        When called update the memory of XMem - Sensory, Working and Long Term
        """
        # Update the hidden state of the sensory memory
        if self.need_segment_flag and self.is_normal_update_flag:
            self.memory.set_hidden(self.hidden)
        self._update_internal_fdx_bis()
        