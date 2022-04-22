# Copyright 2021 Microsoft . All rights reserved.
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
from torch import nn
from towhee.models.utils.general_utils import to_2tuple


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding

    Args:
        img_size (`int=224`):
            image height (should be equal to width)
        patch_size (`int=16`):
            patch height (should be equal to width)
        in_chans (`int=3`):
            the number of image channels
        embed_dim (`int=768`):
            embedding dimension
        norm_layer (`nn.Module=None`):
            normalization layer
        flatten (`bool=True`):
            if flat output

    Example:
        >>> import torch
        >>> from towhee.models.layers.patch_embed2d import PatchEmbed2D
        >>>
        >>> test_shape1 = (1, 3, 224, 224)
        >>> test_shape2 = (1, 3, 5, 224, 224)
        >>> fake_img = torch.rand(test_shape1)
        >>> fake_video = torch.rand(test_shape2)
        >>> model = PatchEmbed2D()
        >>> out1 = model(fake_img)
        >>> out2 = model(fake_video)
        >>> print(out1.shape, out2.shape)
        torch.Size([1, 196, 768]) torch.Size([5, 196, 768])
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 4:
            _, _, h, w = x.shape
            assert h == self.img_size[0] and w == self.img_size[1], \
                f'Input image size ({h}*{w}) doesn\'t match model ({self.img_size[0]}*{self.img_size[1]}).'
        elif len(x.shape) == 5:
            _, _, _, h, w = x.shape
            x = x.permute(1, 3, 4, 0, 2).flatten(3).permute(3, 0, 1, 2)  # BCTHW -> (B*T)CHW
            assert h == self.img_size[0] and w == self.img_size[1], \
                f'Input frame size ({h}*{w}) doesn\'t match model ({self.img_size[0]}*{self.img_size[1]}).'
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# if __name__ == '__main__':
#     import torch
#
#     test_shape1 = (1, 3, 224, 224)
#     test_shape2 = (1, 3, 5, 224, 224)
#     fake_img = torch.rand(test_shape1)
#     fake_video = torch.rand(test_shape2)
#     model = PatchEmbed2D()
#     out1 = model(fake_img)
#     out2 = model(fake_video)
#     assert(out1.shape == (1, 196, 768))
#     assert(out2.shape == (5, 196, 768))
