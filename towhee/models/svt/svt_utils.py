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
from towhee.models.vit.vit_utils import get_configs as vit_configs
from functools import partial
from torch import nn


def get_configs(model_name: str = None):
    if model_name == 'svt_vitb_k400':
        configs = vit_configs('vit_base_16x224')
        configs.update(dict(
            num_frames=8,
            attention_type='divided_space_time',
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=400,
            dropout=0.,
            first_conv='patch_embed.proj',
            classifier='head',
            filter_fn=None,
        ))
    else:
        raise AttributeError(f'Invalid model_name {model_name}.')
    return configs

