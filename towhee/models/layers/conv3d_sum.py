# Copyright 2021 Zilliz and Facebook. All rights reserved.
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


class SumConv3D(nn.Module):
    """
    Sum the outputs of 3d convolutions.
    ::
                            Conv3d, Conv3d, ...,  Conv3d
                                           ↓
                                          Sum
    Args:
        in_channels (`int`):
            number of input channels.
        out_channels (`int`):
            number of output channels produced by the convolution(s).
        kernel_size (`Tuple[_size_3_t]`):
            Tuple of sizes of the convolution kernels.
        stride (`tupple[_size_3_t]`):
            Tuple of strides of the convolutions.
        padding (`Tuple[_size_3_t]`):
            Tuple of paddings added to all three sides of the input.
        padding_mode (`Tuple[string]`):
            Tuple of padding modes for each convs.
            Options include `zeros`, `reflect`, `replicate` or `circular`.
        dilation (`Tuple[_size_3_t]`):
            Tuple of spacings between kernel elements.
        groups (`Tuple[_size_3_t]`):
            Tuple of numbers of blocked connections from input
            channels to output channels.
        bias (`Tuple[bool]`):
            If `True`, adds a learnable bias to the output.
        reduction_method (`str`):
            Options include `sum` and `cat`.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
        padding=None,
        padding_mode=None,
        dilation=None,
        groups=None,
        bias=None,
        reduction_method="sum",
    ) -> None:
        super().__init__()
        assert reduction_method in ("sum", "cat")
        self.reduction_method = reduction_method
        conv_list = []
        for ind in range(len(kernel_size)):
            conv_param = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size[ind],
            }
            if stride is not None and stride[ind] is not None:
                conv_param["stride"] = stride[ind]
            if padding is not None and padding[ind] is not None:
                conv_param["padding"] = padding[ind]
            if dilation is not None and dilation[ind] is not None:
                conv_param["dilation"] = dilation[ind]
            if groups is not None and groups[ind] is not None:
                conv_param["groups"] = groups[ind]
            if bias is not None and bias[ind] is not None:
                conv_param["bias"] = bias[ind]
            if padding_mode is not None and padding_mode[ind] is not None:
                conv_param["padding_mode"] = padding_mode[ind]
            conv_list.append(nn.Conv3d(**conv_param))
        self.convs = nn.ModuleList(conv_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        for ind in range(len(self.convs)):
            output.append(self.convs[ind](x))
        if self.reduction_method == "sum":
            output = torch.stack(output, dim=0).sum(dim=0, keepdim=False)
        elif self.reduction_method == "cat":
            output = torch.cat(output, dim=1)
        return output
