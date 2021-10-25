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

import torch

from torch import nn


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling in Deep Convolutional Networks
    """
    def __init__(self, kernel_size1, kernel_size2, kernel_size3):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size1, stride=1, padding=kernel_size1 // 2)
        self.pool2 = nn.MaxPool2d(kernel_size2, stride=1, padding=kernel_size2 // 2)
        self.pool3 = nn.MaxPool2d(kernel_size3, stride=1, padding=kernel_size3 // 2)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x, x1, x2, x3], dim=1)
