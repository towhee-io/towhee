# Original pytorch implementation by:
# 'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
#       - https://arxiv.org/abs/2010.11929
# 'How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers'
#       - https://arxiv.org/abs/2106.10270
#
# Original code by / Copyright 2020, Ross Wightman.
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


import torch
from torch import nn
from functools import partial
from collections import OrderedDict

from towhee.models.utils.init_vit_weights import init_vit_weights
from towhee.models.layers.vit_block import Block
from towhee.models.layers.patch_embed2d import PatchEmbed2D
from towhee.models.vit.vit_utils import get_configs, load_pretrained_model


class VitModel(nn.Module):
    """
    Vision Transformer Model
    Args:
        name (str): model name
        weights_path (str): path to model weights
        pretrained (bool): if model is pretrained
        img_size (int): image height or width (height=width)
        patch_size (int): patch height or width (height=width)
        in_c (int): number of image channels
        num_classes (int): number of classes
        embed_dim (int): number of features
        depth (int): number of blocks
        num_heads (int): number of heads for Multi-Attention layer
        mlp_ratio (int): mlp ratio
        qkv_bias (bool): if add bias to qkv layer
        qk_scale (float): number to scale qk
        representation_size (int): size of representations
        drop_ratio (float): drop rate of a block
        attn_drop_ratio (float): drop rate of attention layer
        drop_path_ratio (float): drop rate of drop_path layer
        embed_layer: patch embedding layer
        norm_layer: normalization layer
        act_layer: activation layer
    """
    def __init__(self,
                 name=None, weights_path=None, pretrained=False,
                 img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 representation_size=None, drop_ratio=0, attn_drop_ratio=0, drop_path_ratio=0,
                 embed_layer=PatchEmbed2D, norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

        if pretrained:
            configs = get_configs(name)
            load_pretrained_model(self, configs, weights_path=weights_path)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
