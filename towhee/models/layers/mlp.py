# Copyright 2021 Ross Wightman . All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is modified by Zilliz.


from torch import nn

from towhee.models.layers.layers_with_relprop import GELU, Linear, Dropout


class Mlp(nn.Module):
    """
    MLP module w/ dropout and configurable activation layer,
    as used in Vision Transformer, MLP-Mixer and related networks.

    Args:
        - in_features (`int`):
            Number of input features.
        - hidden_features (`int`):
            Number of hidden features.
        - out_features (`int`):
            Number of output features.
        - act_layer (`nn.Module`):
            Activation layer.
        - drop (`float`):
            Dropout rate.

    Example:
        >>> import torch
        >>> from towhee.models.layers.mlp import Mlp
        >>>
        >>> fake_input = torch.rand(1, 4)
        >>> layer = Mlp(in_features=4, hidden_features=6, out_features=8)
        >>> output = layer(fake_input)
        >>> print(output.shape)
        torch.Size([1, 8])
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features=in_features, out_features=hidden_features)  # pylint: disable=unexpected-keyword-arg
        self.act = act_layer()
        self.fc2 = Linear(in_features=hidden_features, out_features=out_features)  # pylint: disable=unexpected-keyword-arg
        self.drop = Dropout(p=drop)  # pylint: disable=unexpected-keyword-arg

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam
