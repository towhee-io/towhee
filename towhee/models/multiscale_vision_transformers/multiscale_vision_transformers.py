# Copyright 2021 Zilliz and Facebook. All rights reserved.
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

from towhee.models.utils.init_vit_weights import init_vit_weights


class MultiscaleVisionTransformers(nn.Module):
    """
    Multiscale Vision Transformers
    https://arxiv.org/abs/2104.11227
    ::
                                       PatchEmbed
                                           ↓
                                   PositionalEncoding
                                           ↓
                                        Dropout
                                           ↓
                                     Normalization
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
                                     Normalization
                                           ↓
                                          Head
    Args:
        patch_embed ('nn.Module'):
            Patch embed module.
        cls_positional_encoding ('nn.Module'):
            Positional encoding module.
        pos_drop ('nn.Module'):
            Dropout module after patch embed.
        norm_patch_embed ('nn.Module'):
            Normalization module after patch embed.
        blocks ('nn.ModuleList'):
            Stack of multi-scale transformer blocks.
        norm_embed ('nn.Module'):
            Normalization layer before head.
        head ('nn.Module'):
            Head module.
    """

    def __init__(
        self,
        *,
        patch_embed,
        cls_positional_encoding,
        pos_drop,
        norm_patch_embed,
        blocks,
        norm_embed,
        head,
    ) -> None:
        super().__init__()
        self.patch_embed: nn.Module = patch_embed
        self.cls_positional_encoding: nn.Module = cls_positional_encoding
        self.pos_drop: nn.Module = pos_drop
        self.norm_patch_embed: nn.ModuleList = norm_patch_embed
        self.blocks: nn.ModuleList = blocks
        self.norm_embed: nn.Module = norm_embed
        self.head: nn.Module = head

        assert hasattr(
            cls_positional_encoding, "patch_embed_shape"
        ), "cls_positional_encoding should have attribute patch_embed_shape."
        init_vit_weights(self, trunc_normal_std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_embed is not None:
            x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)

        if self.pos_drop is not None:
            x = self.pos_drop(x)

        if self.norm_patch_embed is not None:
            x = self.norm_patch_embed(x)

        thw = self.cls_positional_encoding.patch_embed_shape
        for blk in self.blocks:
            x, thw = blk(x, thw)
        if self.norm_embed is not None:
            x = self.norm_embed(x)
        if self.head is not None:
            x = self.head(x)
        return x
