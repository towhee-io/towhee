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
from torch import nn

def conv_bn(inp: int, oup: int, stride: int = 1, leaky: int = 0) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp: int, oup: int, stride: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1x1(inp: int, oup: int, stride: int, leaky: float=0) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.

    Args:
        loc (`torch.FloatTensor`):
            location predictions for loc layers, Shape: [num_priors,4].
        priors (`torch.FloatTensor`):
            Prior boxes in center-offset form. Shape: [num_priors,4].
        variances (`list[float]`):
            Variances of priorboxes.
    Return:
        (`torch.FloatTensor`)
            decoded bounding box predictions.
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """
    Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.

    Args:
        pre (`torch.FloatTensor`):
            landm predictions for loc layers, Shape: [num_priors,10].
        priors (`torch.FloatTensor`):
            Prior boxes in center-offset form. Shape: [num_priors,4].
        variances (`list[float]`):
            Variances of priorboxes.
    Return:
        (`torch.FloatTensor`)
            decoded landm predictions.
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms
