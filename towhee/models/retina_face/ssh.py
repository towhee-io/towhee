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
import torch
import torch.nn.functional as F
from torch import nn

from towhee.models.retina_face.utils import conv_bn, conv_bn_no_relu

class SSH(nn.Module):
    """
    SSH Module

    Single stage headless Module inspired by SSH: Single Stage Headless Face Detector.
    Described in: https://arxiv.org/abs/1708.03979.

    Args:
        in_channel (`int`):
            number of input channels.
        out_channel (`int`):
            number of output channels.
    """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3x3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5x5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5x5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7x7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, x: torch.FloatTensor):
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv7x7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = F.relu(out)
        return out
