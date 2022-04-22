# Built on top of codes from / Copyright 2020 Ross Wightman & Copyright (c) Facebook, Inc. and its affiliates.
#
# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from functools import partial
from collections import OrderedDict

import torch
from torch import nn
from torch.utils import model_zoo

from towhee.models.vit.vit_utils import get_configs as vit_configs

log = logging.getLogger()


def map_state_dict(checkpoint, use_ema=False):
    state_dict_key = 'state_dict'
    if isinstance(checkpoint, dict):
        if use_ema and 'state_dict_ema' in checkpoint:
            state_dict_key = 'state_dict_ema'
    if state_dict_key and state_dict_key in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint[state_dict_key].items():
            # strip `module.` prefix
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif 'model_state' in checkpoint:
        state_dict_key = 'model_state'
        new_state_dict = OrderedDict()
        for k, v in checkpoint[state_dict_key].items():
            # strip `model.` prefix
            name = k[6:] if k.startswith('model') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        state_dict = checkpoint
    log.info('New state_dict keys: %s', state_dict_key)
    return state_dict


def load_pretrained(
        model,
        cfg,
        checkpoint_path=None,
        strict=True,
        device='cpu'
):

    num_classes = cfg['num_classes']
    in_c = cfg['in_c']
    img_size = cfg['img_size']
    patch_size = cfg['patch_size']
    filter_fn = cfg['filter_fn']
    num_frames = cfg['num_frames']
    # attention_type = cfg['attention_type']

    if checkpoint_path is None:
        if 'url' not in cfg or cfg['url'] is None:
            log.error('No pretrained weights are provided.')
            raise AttributeError('No pretrained weights are provided.')
        else:
            state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = torch.load(checkpoint_path, map_location=device)['model']

    state_dict = map_state_dict(checkpoint=state_dict)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_c == 1:
        conv1_name = cfg['first_conv']
        log.info('Converting first conv %s pretrained weights from 3 to 1 channel.', conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        shape_o, shape_i, shape_j, shape_k = conv1_weight.shape
        if shape_i > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(shape_o, shape_i // 3, 3, shape_j, shape_k)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_c != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        shape_o, shape_i, shape_j, shape_k = conv1_weight.shape
        if shape_i != 3:
            log.warning('Deleting first conv %s from pretrained weights.', conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            log.info('Repeating first conv %s weights in channel dim.', conv1_name)
            repeat = int(math.ceil(in_c / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_c, :, :]
            conv1_weight *= (3 / float(in_c))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    # Resize the positional embeddings in case they don't match
    num_patches = (img_size // patch_size) * (img_size // patch_size)
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = nn.functional.interpolate(other_pos_embed, size=num_patches, mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed

    # Resize time embeddings in case they don't match
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = nn.functional.interpolate(time_embed, size=num_frames, mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)

    # Initialize temporal attention
    # if attention_type == 'divided_space_time':
    #     new_state_dict = state_dict.copy()
    #     for key in state_dict:
    #         if 'blocks' in key and 'attn' in key:
    #             new_key = key.replace('attn', 'temporal_attn')
    #             if new_key not in state_dict:
    #                 new_state_dict[new_key] = state_dict[key]
    #             else:
    #                 new_state_dict[new_key] = state_dict[new_key]
    #         if 'blocks' in key and 'norm1' in key:
    #             new_key = key.replace('norm1', 'temporal_norm1')
    #             if new_key not in state_dict:
    #                 new_state_dict[new_key] = state_dict[key]
    #             else:
    #                 new_state_dict[new_key] = state_dict[new_key]
    #     state_dict = new_state_dict

    # Load the weights
    model.load_state_dict(state_dict, strict=strict)
    return model


def get_configs(model_name):
    if model_name == 'timesformer_k400_8x224':
        configs = vit_configs('vit_base_16x224')
        configs.update(dict(
            url='https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0',
            num_frames=8,
            attention_type='divided_space_time',
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=400,
            dropout=0.,
            first_conv='patch_embed.proj',
            classifier='head',
            filter_fn=None,
        ))
        return configs
