# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
