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
# Original code from https://github.com/jhcho99/CoFormer.
#
# Modified by Zilliz.

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from towhee.models.coformer.utils import NestedTensor, is_main_process
from towhee.models.layers.position_encoding import build_position_encoding


class BackboneBase(nn.Module):
    """
    Backbone base block.

    Args:
        backbone(`nn.Module`):
            Backbone model.
        train_backbone(`bool`):
            If backbone trained.
        name_backbone(`str`):
            Name of the backbone.
        num_channels(`int`):
            The number of the channels.
        return_interm_layers(`bool`):
            If the model returns intermediate layers.
    """
    def __init__(self,
                 backbone: nn.Module,
                 train_backbone: bool,
                 name_backbone: str,
                 num_channels: int,
                 return_interm_layers: bool
                ) -> None:
        super().__init__()
        if "resnet" in name_backbone:
            if not train_backbone:
                for _, parameter in backbone.named_parameters():
                    parameter.requires_grad_(False)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {"layer4": "0"}
        else:
            # only resnet50 is supported
            assert False, f"backbone {name_backbone} is not supported now"
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm.

    Args:
        name (`str`):
            Backbone name.
        train_backbone (`bool`):
            The flag to control train mode for backbone.
        return_interm_layers (`bool`):
            The flag to return intermediate layers.
        dilation (`bool`):
            The flag to use dilated model.
    """
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if "resnet" in name:
            backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                         pretrained=is_main_process())
            num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        else:
            # TODO only resnet is supported
            assert False, f"backbone {name} is not supported now"
        super().__init__(backbone, train_backbone, name, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """
    Joiner class.

    Args:
        backbone (`Backbone`):
            ResNet backbone with frozen BatchNorm.
        position_embedding (`nn.Module`):
            Position embedding layer.
    """
    # pylint: disable=W0235
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # pylint: disable=W0237
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []

        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(
                    hidden_dim=512,
                    position_embedding="learned",
                    backbone="resnet50",
                  ):
    position_embedding = build_position_encoding(
                                                 hidden_dim=hidden_dim,
                                                 position_embedding=position_embedding,
                                                )
    backbone = Backbone(backbone, train_backbone=False, return_interm_layers=False, dilation=False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
