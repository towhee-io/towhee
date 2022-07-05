# Reference:
#   [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
#
# Built on top of codes from / Copyright (c) Facebook, Inc. and its affiliates.
# Modifications & additions by / Copyright 2021 Zilliz. All rights reserved.
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

import os
import torch
from torch import nn

try:
    from einops import rearrange
except ModuleNotFoundError:
    os.system('pip install einops')
    from einops import rearrange

from towhee.models.layers.attention import MultiHeadAttention
from towhee.models.layers.droppath import DropPath
from towhee.models.layers.mlp import Mlp


class Block(nn.Module):
    """
    TimeSformer block.

    Args:
        dim (`int`):
            feature dimension
        num_heads (`int`):
            number of attention heads
        mlp_ratio (`float`):
            mlp ratio
        qkv_bias (`bool=False`):
            if use qkv_bias
        qk_scale (`float=None`):
            number to scale qk
        drop (`float=0.`):
            drop rate of projection in attention & MLP layers
        attn_drop (`float=0.`):
            attention drop rate
        drop_path (`float=0.1`):
            drop probability in DropPath
        act_layer (`nn.module=nn.GELU`):
            module used in activation layer
        norm_layer (`nn.module=nn.LayerNorm`):
            module used in normalization layer
        attention_type (`str='divided_space_time`):
            type of TimeSformer attention from ['divided_space_time', 'space_only', 'joint_space_time']

    Example:
        >>> import torch
        >>> from towhee.models.timesformer.timesformer_block import Block
        >>>
        >>> test_shape = (1, 196*8+1, 768)
        >>> fake_x = torch.rand(test_shape)
        >>> model = Block(dim=768, num_heads=12)
        >>> out = model(fake_x, b=1, t=8, w=int(224/16))
        >>> print(out.shape)
        torch.Size([1, 1569, 768])
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only', 'joint_space_time', 'frozen_in_time'])

        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_ratio=attn_drop, proj_drop_ratio=drop)

        # Temporal Attention Parameters
        if self.attention_type in ['divided_space_time', 'frozen_in_time']:
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = MultiHeadAttention(
                dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop_ratio=attn_drop, proj_drop_ratio=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, b, t, w):
        num_spatial_tokens = (x.size(1) - 1) // t
        h = num_spatial_tokens // w

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type in ['divided_space_time', 'frozen_in_time'] :
            # Temporal
            original_input = x[:, 1:, :]
            xt = x[:, 1:, :]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=b, h=h, w=w, t=t)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=b, h=h, w=w, t=t)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal

            # Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, t, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=b, t=t).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=b, h=h, w=w, t=t)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            # CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=b, t=t)
            cls_token = torch.mean(cls_token, 1, True)  # averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=b, h=h, w=w, t=t)
            res = res_spatial
            x = xt

            # Mlp
            if self.attention_type == 'frozen_in_time':
                x = torch.cat((init_cls_token, original_input), 1) + torch.cat((cls_token, res), 1)

            else:
                x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x



# if __name__ == '__main__':
#     import torch
#
#     # batch=1, num_frames=8, img_size=224*224, patch_size=16*16
#     test_shape = (1, 196*8+1, 768)
#     fake_x = torch.rand(test_shape)
#     model = Block(dim=768, num_heads=12)
#     out = model(fake_x, b=1, t=8, w=int(224/16))
#     assert(out.shape == test_shape)
