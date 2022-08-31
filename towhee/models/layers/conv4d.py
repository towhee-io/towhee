# Original pytorch implementation by:
# 'Correlation Verification for Image Retrieval'
#       - https://arxiv.org/abs/2204.01458
# Original code by / Copyright 2022, Seongwon Lee.
# Modifications & additions by / Copyright 2022 Zilliz. All rights reserved.
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


class CenterPivotConv4d(nn.Module):
    """
    CenterPivot 4D conv

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`list or tuple`):
            Numbers for kernel size.
        stride (`list or tuple`):
            Numbers for stride.
        padding (`list or tuple`):
            Numbers for padding.
        bias (`bool`):
            Whether to add bias for conv layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False
        self.idx_initialized_2 = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
        idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
        self.len_h = len(idxh)
        self.len_w = len(idxw)
        self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
        self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def prune_out2(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        idxh = torch.arange(start=0, end=ha, step=self.stride[:2][0], device=ct.device)
        idxw = torch.arange(start=0, end=wa, step=self.stride[:2][1], device=ct.device)
        self.len_h = len(idxh)
        self.len_w = len(idxw)
        self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wa).view(-1)
        self.idx_initialized_2 = True
        ct_pruned = ct.view(bsz, ch, -1, hb, wb).permute(0, 1, 3, 4, 2).index_select(4, self.idx).\
            permute(0, 1, 4, 2, 3).view(bsz, ch, self.len_h, self.len_w, hb, wb)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        if self.stride[:2][-1] > 1:
            out2 = self.prune_out2(x)
        else:
            out2 = x
        bsz, inch, ha, wa, hb, wb = out2.size()
        out2 = out2.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()
        y = out1 + out2
        return y
