# Pytorch implementation of HorNet from
#   [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions]
#   (https://arxiv.org/abs/2207.14284).
#
# Inspired by https://github.com/raoyongming/HorNet
#
# Modifications & additions by Copyright 2021 Zilliz. All rights reserved.
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

from towhee.models.utils.weight_init import trunc_normal_
from towhee.models.utils.download import download_from_url
from towhee.models.convnext.utils import LayerNorm
from towhee.models.hornet import Block, GatedConv, get_configs


class HorNet(nn.Module):
    """
    HorNet

    Args:
        in_chans (`int`): Number of input channels.
        num_classes (`int`): Number for classes for classification.
        depths (`tuple`): Model depths.
        base_dim (`int`): Base dimensions.
        drop_path_rate (`float`): Drop ratio of drop path.
        layer_scale_init_value (`float`): Initial value to scale layer.
        head_init_scale (`float`): Initial value to scale head.
        gnconv (`nn.Module`): gnConv layer(s).
        uniform_init (`bool`): Flag to control whether to apply the uniform initial weights.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 9, 3), base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gnconv=GatedConv, block=Block, uniform_init=False
                 ):
        super().__init__()

        self.base_dim = base_dim
        self.num_classes = num_classes

        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        assert len(gnconv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for blk in self.stages[i]:
                x = blk(x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        weights_path: str = None,
        device: str = None,
        **kwargs,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = get_configs(model_name)
    configs.update(**kwargs)
    if 'url' in configs:
        url = configs['url']
        del configs['url']
    else:
        url = None

    model = HorNet(**configs).to(device)

    if pretrained:
        if weights_path is None:
            assert url, 'No default url or weights path is provided for the pretrained model.'
            weights_path = download_from_url(url)
        state_dict = torch.load(weights_path, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    model.eval()
    return model
