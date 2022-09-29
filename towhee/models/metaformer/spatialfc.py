# Built on top of the original implementation at https://github.com/sail-sg/poolformer/blob/main/models/metaformer.py
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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

from torch import nn
import torch
from typing import Sequence
from functools import partial, reduce


class SpatialFc(nn.Module):
    """
    SpatialFc module that take features with shape of (B,C,*) as input.
    Args:
        spatial_shape (tuple[int]): spatial shape
    """
    def __init__(self, spatial_shape=[14, 14]):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        n = reduce(lambda x, y: x * y, spatial_shape)
        self.fc = nn.Linear(n, n, bias=False)

    def forward(self, x):
        # input shape like [B, C, H, W]
        shape = x.shape
        x = torch.flatten(x, start_dim=2)  # [B, C, H*W]
        x = self.fc(x)  # [B, C, H*W]
        x = x.reshape(*shape)  # [B, C, H, W]
        return x
