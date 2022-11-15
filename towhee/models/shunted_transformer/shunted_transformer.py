# Pytorch implementation of Shunted Transform in the paper:
#   [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/abs/2111.15193)
#
# Inspired by original code from https://github.com/OliverRensu/Shunted-Transformer
#
# Additions & modifications are protected by Copyright 2021 Zilliz. All rights reserved.
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

import torch
from torch import nn

from towhee.models.utils import create_model as towhee_model
from towhee.models.utils.init_vit_weights import init_vit_weights
from towhee.models.shunted_transformer import OverlapPatchEmbed, HeadPatchEmbed, Block, get_configs


class ShuntedTransformer(nn.Module):
    """
    Shunted Transformer.

    Args:
        img_size (`int`): image size (width or height)
        in_chans (`int`): input channels
        num_classes (`int`): number of classes for classification
        embed_dims (`List[int]`): a list of embedding dimensions at different stages
        num_heads (`List[int]`): a list of attention heads at different stages
        mlp_ratios (`List[int]`): a list of mlp ratios at different stages
        qkv_bias (`bool`): flag to enable bias for qkv
        qk_scale (`int`): number to scale qk
        drop_rate (`float`): drop rate of mlp layers
        attn_drop_rate (`float`): drop rate of attention
        drop_path_rate (`float`): drop rate of drop path
        norm_layer (`nn.Module`): normalization layer
        depths (`List[int]`): a list of stochastic depths for blocks
        sr_ratios (`List[int]`): a list of shunt rates in blocks at each stage
        num_stages (`int`): number of stages
        num_conv (`int`): number of additional convolutions in patch embedding layer at the first stage
    """
    def __init__(self, img_size=224, num_classes=1000, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=None, sr_ratios=None, num_stages=4, num_conv=0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.sr_ratios = sr_ratios
        self.depths = depths
        self.num_stages = num_stages
        self.num_conv = num_conv

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = HeadPatchEmbed(self.num_conv, embed_dim=self.embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)),
                                                patch_size=3,
                                                stride=2,
                                                in_chans=self.embed_dims[i - 1],
                                                embed_dim=self.embed_dims[i])

            block = nn.ModuleList([Block(
                dim=self.embed_dims[i], num_heads=self.num_heads[i], mlp_ratio=self.mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=self.sr_ratios[i])
                for j in range(self.depths[i])])
            norm = norm_layer(self.embed_dims[i])
            cur += self.depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        # classification head
        self.head = nn.Linear(self.embed_dims[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(init_vit_weights)

    def forward_features(self, x):
        b = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, h, w = patch_embed(x)
            for blk in block:
                x = blk(x, h, w)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None,
        **kwargs
        ):
    configs = get_configs(model_name)
    configs.update(**kwargs)

    model = towhee_model(ShuntedTransformer, configs=configs, pretrained=pretrained, checkpoint_path=checkpoint_path, device=device)
    return model
