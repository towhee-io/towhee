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

from torch import nn

from towhee.models.layers.patch_embed2d import PatchEmbed2D
from towhee.models.layers.mlp import Mlp as ViTMlp
from towhee.models.utils.init_vit_weights import init_vit_weights


class OverlapPatchEmbed(PatchEmbed2D):
    """
    Overlap patch embedding layer based on a standard patch embedding layer in ViT

    Args:
        img_size (`int`): image size (width or height)
        patch_size (`int`): patch size
        stride (`int`): stride used in convolution
        in_chans (`int`): number of input channels
        embed_dim (`int`): output dimension
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=nn.LayerNorm, flatten=True)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.apply(init_vit_weights)


class Mlp(ViTMlp):
    """
    Modified Mlp containing depthwise convolution, based on a standard Mlp in ViT.

    Args:
        in_features (`int`): input dimension
        hidden_features (`int`): dimension used in depthwise conv layer
        out_features (`int`): output dimension
        act_layer (`nn.Module`): activation layer
        drop (`float`): dropout rate
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop
        )
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, _, c = x.shape

        x0 = x.transpose(1, 2).view(b, c, h, w)
        x0 = self.dwconv(x0)
        x0 = x0.flatten(2).transpose(1, 2)

        x = self.act(x + x0)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
