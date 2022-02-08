# Copyright 2021 biubug6 . All rights reserved.
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

#adapted from https://github.com/biubug6/Pytorch_Retinaface
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from towhee.models.retina_face.utils import conv_bn1x1, conv_bn

class RetinaFaceFPN(nn.Module):
    """
    RetinaFaceFPN

    Feature Pyramid Network for RetinaFace.

    Args:
        in_channels_list (`List[int]`):
            input channel numbers ofr each FPN layer.
        out_channels (`int`):
            output channel number.
    """
    def __init__(self,in_channels_list: List[int],out_channels:int):
        super().__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, x: torch.FloatTensor):
        # names = list(input.keys())
        x = list(x.values())

        output1 = self.output1(x[0])
        output2 = self.output2(x[1])
        output3 = self.output3(x[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out
