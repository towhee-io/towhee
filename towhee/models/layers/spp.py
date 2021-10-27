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
import math
import torch
from torch import nn


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling in Deep Convolutional Networks
    pre_conv: a tensor vector of previous convolution layer
    pre_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    def __init__(self, pre_conv, pre_conv_size, out_pool_size):
        super().__init__()
        sample_num = pre_conv.shape[0]
        for i in range(len(out_pool_size)):
            height = int(math.ceil(pre_conv_size[0] / out_pool_size[i]))
            width = int(math.ceil(pre_conv_size[1] / out_pool_size[i]))
            h_pad = (height * out_pool_size[i] - pre_conv_size[0] + 1) / 2
            w_pad = (width * out_pool_size[i] - pre_conv_size[1] + 1) / 2
            maxpool = torch.nn.MaxPool2d((height, width), stride=(height, width), padding=(int(h_pad), int(w_pad)))
            x = maxpool(pre_conv)
            if (i == 0):
                self.spp = x.view(sample_num, -1)
            else:
                self.spp = torch.cat((self.spp, x.view(sample_num, -1)), 1)

    def forward(self):
        return self.spp
