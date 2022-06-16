# Copyright 2022 Zilliz. All rights reserved.
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
#
# Code inspired by https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html
#
# Original code from https://github.com/Atze00/MoViNet-pytorch
#
# Modified by Zilliz.

from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from towhee.models.movinet.movinet_block import ConvBlock3D, BasicBneck
from towhee.models.utils.causal_module import CausalModule
from towhee.models.layers.activations import Swish
from towhee.models.layers.temporal_cg_avgpool3d import TemporalCGAvgPool3D
from towhee.models.movinet.config import _C

class MoViNet(nn.Module):
    """
    Args:
        causal(`string`):
            Causal mode.
        pretrained(`bool`):
            Pretrained models. If pretrained is True, num_classes is set to 600,
            conv_type is set to "3d" if causal is False, "2plus1d" if causal is True
            tf_like is set to True.
        num_classes(`int`):
            Number of classes for classifcation.
        conv_type(`string`):
            Type of convolution either 3d or 2plus1d
        tf_like(`bool`):
            Tf_like behaviour, basically same padding for convolutions.
    """
    def __init__(self,
                 cfg: "CfgNode",
                 causal: bool = True,
                 pretrained: bool = False,
                 num_classes: int = 600,
                 conv_type: str = "3d",
                 tf_like: bool = False
                 ) -> None:
        super().__init__()
        if pretrained:
            tf_like = True
            num_classes = 600
            conv_type = "2plus1d" if causal else "3d"
        blocks_dic = OrderedDict()

        norm_layer = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = Swish if conv_type == "3d" else nn.Hardswish

        # conv1
        self.conv1 = ConvBlock3D(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
            )
        # blocks
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(basicblock,
                                                      causal=causal,
                                                      conv_type=conv_type,
                                                      tf_like=tf_like,
                                                      norm_layer=norm_layer,
                                                      activation_layer=activation_layer
                                                      )
        self.blocks = nn.Sequential(blocks_dic)
        # conv7
        self.conv7 = ConvBlock3D(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
            )
        # pool
        self.classifier = nn.Sequential(
            # dense9
            ConvBlock3D(cfg.conv7.out_channels,
                        cfg.dense9.hidden_dim,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias=True),
            Swish(),
            nn.Dropout(p=0.2, inplace=True),
            # dense10d
            ConvBlock3D(cfg.dense9.hidden_dim,
                        num_classes,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias=True),
        )
        if causal:
            self.cgap = TemporalCGAvgPool3D()
        if pretrained:
            if causal:
                if cfg.name not in ["A0", "A1", "A2"]:
                    raise ValueError("Only A0,A1,A2 streaming" +
                                     "networks are available pretrained")
                state_dict = (torch.hub
                              .load_state_dict_from_url(cfg.stream_weights))
            else:
                state_dict = torch.hub.load_state_dict_from_url(cfg.weights)
            self.load_state_dict(state_dict)
        else:
            self.apply(self._weight_init)
        self.causal = causal

    def avg(self, x: Tensor) -> Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        return x

    def head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        x = x.flatten(1)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)

def create_model(
        model_name: str = "movineta0",
        pretrained: bool = False,
        causal: bool = False,
        device: str = "cpu",
        ):
    if model_name == "movineta0":
        model_config = _C.MODEL.MoViNetA0
    elif model_name == "movineta1":
        model_config = _C.MODEL.MoViNetA1
    elif model_name == "movineta2":
        model_config = _C.MODEL.MoViNetA2
    elif model_name == "movineta3":
        model_config = _C.MODEL.MoViNetA3
    elif model_name == "movineta4":
        model_config = _C.MODEL.MoViNetA4
    elif model_name == "movineta5":
        model_config = _C.MODEL.MoViNetA5
    else:
        raise AttributeError(f"Invalid model_name {model_name}.")
    model = MoViNet(
                    cfg = model_config,
                    causal = causal,
                    pretrained = pretrained,
                    num_classes = 600,
                    conv_type = "3d",
                    tf_like = False
                    )
    model.to(device)
    return model
