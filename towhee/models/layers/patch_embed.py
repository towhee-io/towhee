# Copyright 2021 Ross Wightman and Zilliz. All rights reserved.
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

class PatchEmbed(nn.Module):
    """
    Get patch-embeddings of ViT; Convert 2D image to patch embeddings.

    Args:
    img_size: height or width of the input image
    patch_size: height or width of a patch
    in_c: number of channels of the input image, eg. (R, G, B) means 3 channels
    embed_dim: length of embedding for each patch
    norm_layer: boolean value, if or not add norm_layer
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, img_height, img_width = x.shape
        assert img_height == self.img_size[0] and img_width == self.img_size[1], \
            f'Unmatched image size: expected ({self.img_size}) but received ({img_height}, {img_width}).'

        x = self.proj(x).flatten(2).transpose(1,2)
        x = self.norm(x)
        return x
