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

class ClassHead(nn.Module):
    """
    ClassHead

    RetinaFace head for classification branch.

    Args:
        inchannels (`int`):
            number of input channels.
        num_anchors (`int`):
            number of anchors.
    """
    def __init__(self,inchannels: int=512,num_anchors: int=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x: torch.FloatTensor):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    """
    BboxHead

    RetinaFace head for bounding box branch.

    Args:
        inchannels (`int`):
            number of input channels.
        num_anchors (`int`):
            number of anchors.
    """
    def __init__(self,inchannels: int=512,num_anchors: int=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x: torch.FloatTensor):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    """
    LandmarkHead

    RetinaFace head for landmark branch.

        inchannels (`int`):
            number of input channels.
        num_anchors (`int`):
            number of anchors.
    """
    def __init__(self,inchannels: int=512,num_anchors: int=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x: torch.FloatTensor):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)
