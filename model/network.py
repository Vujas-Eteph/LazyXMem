"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn

from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *

from icecream import ic


class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')

        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)

        # Projection from f16 feature space to key/value space
        self.key_proj = KeyProjection(1024, self.key_dim)

        self.decoder = Decoder(self.value_dim, self.hidden_dim)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)
            
        self.pmf_per_obj = None

    def encode_key(self, frame, need_sk=True, need_ek=True): 
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
    
        f16, f8, f4 = self.key_encoder(frame)
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks,
                     is_deep_update=True):
        '''
        This code snippet is designed to process a batch of binary masks
        representing different objects within images. The 'masks' tensor has a 
        shape where the first dimension likely represents different instances (e.g., images), 
        and the second dimension indexes the objects within each image.
        '''
        num_objects = masks.shape[1]
        if num_objects != 1:
            # We iterate over each object and sum the masks of all other objects,
            # excluding the current one (indexed by i). The summed masks are 
            # concatenated into a tensor 'others', which has the same second 
            # dimension as 'masks'. This results in a tensor where each column
            # corresponds to the combined mask of all objects except one.
            others = torch.cat([torch.sum(masks[:, [j for j in range(num_objects) if i!=j]], dim=1, keepdim=True) for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        ic(others.shape)
        # other is of shape [batch size, #objects, H, W]
        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, 
                                      others, is_deep_update)
        # g16 [B, n, C_v, H/16, W/16]
        # h16 [B, n, C_h, H/16, W/16]

        return g16, h16

    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key,
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def segment(self, multi_scale_features, memory_readout,
                hidden_state, selector=None, h_out=True, strip_bg=True):

        self.pmf_per_obj = None
        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
            
        self.pmf_per_obj = prob.clone()
            
        # TODO: If I want to save something, i save it here, for the pseudo TTA
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0]//3
            print(f'Hyperparameters read from the model weights: '
                    f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict)
