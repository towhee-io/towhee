# Reference:
#   [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
#
# Built on top of codes from / Copyright 2020 Ross Wightman & Copyright (c) Facebook, Inc. and its affiliates.
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

from towhee.models.utils.init_vit_weights import init_vit_weights
from towhee.models.layers.patch_embed2d import PatchEmbed2D
from towhee.models.timesformer.timesformer_block import Block
from towhee.models.timesformer.timesformer_utils import load_pretrained, get_configs


class TimeSformer(nn.Module):
    """
    TimeSformer model.

    Args:
        img_size (`int=224`):
            image height of video frame (equal to width)
        patch_size (`int=16`):
            patch height (equal to width)
        in_c (`int=3`):
            number of image channel
        num_classes (`int=1000`):
            number of categories of classification
        embed_dim (`int=768`):
            number of hidden features
        depth (`int=12`):
            number of blocks in model
        num_heads (`int`):
            number of attention heads
        mlp_ratio (`float`):
            mlp ratio
        qkv_bias (`bool=False`):
            if use qkv_bias
        qk_scale (`float=None`):
            number to scale qk
        drop_ratio (`float=0.`):
            drop rate of blocks & position embedding layer
        attn_drop_ratio (`float=0.`):
            attention drop rate
        norm_layer (`nn.module=nn.LayerNorm`):
            module used in normalization layer
        num_frames (`int=8`):
            number of samples to take frames
        attention_type (`str='divided_space_time`):
            type of TimeSformer attention from ['divided_space_time', 'space_only', 'joint_space_time']
        dropout (`float=0.`):
            drop ratio of Dropout layer

    Examples:
        >>> import torch
        >>> from towhee.models.timesformer.timesformer import TimeSformer
        >>>
        >>> fake_video = torch.randn(1, 3, 8, 224, 224)  # (batch x channels x frames x height x width)
        >>> model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time')
        >>> pred = model(fake_video)
        >>> print(pred.shape)
        torch.Size([1, 400])
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0.1, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time', dropout=0.):
        super().__init__()
        assert (attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])
        self.img_size = img_size
        self.patch_size = patch_size
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_c, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_ratio)

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_ratio, attn_drop=attn_drop_ratio, drop_path=dpr[i], norm_layer=norm_layer,
                attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(init_vit_weights)

        # Initialize attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def forward_features(self, x):
        b, _, t, h, _ = x.shape
        w = self.num_patches
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            p = int(other_pos_embed.size(2) ** 0.5)
            h = x.size(1) // w
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), p, p)
            new_pos_embed = nn.functional.interpolate(other_pos_embed, size=(h, w), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:b, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=b, t=t)
            # Resize time embeddings in case they don't match
            if t != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = nn.functional.interpolate(time_embed, size=(t), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=b, t=t)
            x = torch.cat((cls_tokens, x), dim=1)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x, b, t, w)

        # Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=b, t=t)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def timesformer(
        model_name: str,
        pretrained: bool = True,
        checkpoint_path: str = None,
        device: str = 'cpu'
):
    if device != 'cpu':
        assert torch.cuda.is_available()

    configs = get_configs(model_name)

    model = TimeSformer(
        img_size=configs['img_size'],
        patch_size=configs['patch_size'],
        in_c=configs['in_c'],
        num_classes=configs['num_classes'],
        embed_dim=configs['embed_dim'],
        depth=configs['depth'],
        num_heads=configs['num_heads'],
        mlp_ratio=configs['mlp_ratio'],
        qkv_bias=configs['qkv_bias'],
        qk_scale=configs['qk_scale'],
        drop_ratio=configs['drop_ratio'],
        attn_drop_ratio=configs['attn_drop_ratio'],
        drop_path_ratio=configs['drop_path_ratio'],
        norm_layer=configs['norm_layer'],
        num_frames=configs['num_frames'],
        attention_type=configs['attention_type'],
        dropout=configs['dropout']
    )

    model.to(device)
    if pretrained:
        model = load_pretrained(model, configs, checkpoint_path=checkpoint_path, strict=True, device=device)

    model.eval()
    return model


# if __name__ == '__main__':
#     import torch
#     import random
#     import numpy as np
#
#     def setup_seed(seed):
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         np.random.seed(seed)
#         random.seed(seed)
#         torch.backends.cudnn.deterministic = True
#
#     setup_seed(24)
#     dummy_video = torch.randn(1, 3, 8, 224, 224)  # (batch x channels x frames x height x width)
#
#     # model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='joint_space_time')
#     # pred = model(dummy_video)
#     # assert(pred.shape == (1, 400))
#
#     model = timesformer(model_name='timesformer_k400_8x224')
#     print(model(dummy_video))
