# Copyright 2021  Facebook. All rights reserved.
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
# This code is modified by Zilliz.


import torch
from torch import nn


class SpatialTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatial-temporal encoding to a tensor.
    Args:
        embed_dim(int):
            Embedding dimension for input sequence.
        patch_embed_shape(Tuple):
            The number of patches in each dimension (T, H, W) after patch embedding.
        sep_pos_embed(bool):
            If set to true, one positional encoding is used for
            spatial patches and another positional encoding is used for temporal
            sequence. Otherwise, only one positional encoding is used for all the patches.
        has_cls(bool):
            If set to true, a cls token is added in the beginning of each input sequence.
    """

    def __init__(
        self,
        embed_dim,
        patch_embed_shape,
        sep_pos_embed=False,
        has_cls=True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed_shape = patch_embed_shape
        self.sep_pos_embed = sep_pos_embed
        self.has_cls = has_cls
        assert (
            len(self.patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.has_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patch, self.embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, self.embed_dim)
            )
            if self.has_cls:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

    def get_patch_embed_shape(self):
        return self.patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor):
                Input tensor.
        """
        b, _, _ = x.shape
        if self.has_cls:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            if self.has_cls:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x
