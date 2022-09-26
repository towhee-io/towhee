# Built on top of the original implementation at https://github.com/mesnico/Wiki-Image-Caption-Matching/blob/master/mcprop/model.py
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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


class DepthAggregator(nn.Module):
    """
    Depth aggregator
    Args:
        aggr (str): aggregator
        input_dim (int): input dimension
        output_dim (int): output dimension
    """
    def __init__(self, aggr, input_dim=1024, output_dim=1024):
        super().__init__()
        self.aggr = aggr
        if self.aggr == 'gated':
            self.self_attn = nn.MultiheadAttention(input_dim, num_heads=4, dropout=0.1)
            self.gate_ffn = nn.Linear(input_dim, 1)

        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, x, mask):
        """
        Forward function
        Args:
            x (torch.tensor): tensor with shape of (depth, B, N, dim)
            mask (torch.tensor): mask of x
        :return
            tensor with shape of (B, dim)
        """
        if self.aggr is None:
            out = x[-1, :, 0, :]  # simply takes the cls token from the last layer
        elif self.aggr == 'mean':
            out = x[:, :, 0, :].mean(dim=0)  # average the cls token from the last layer
        elif self.aggr == 'gated':
            mask_bool = mask.clone()
            mask_bool = mask_bool.bool()
            mask_bool = ~mask_bool
            mask_bool = mask_bool.unsqueeze(1).expand(-1, x.shape[0], -1)
            mask_bool = mask_bool.reshape(-1, mask_bool.shape[2])

            orig = x
            bs = x.shape[1]
            # merge batch size and depth
            x = x.view(-1, x.shape[2], x.shape[3]).permute(1, 0, 2)
            sa, _ = self.self_attn(x, x, x, key_padding_mask=mask_bool)
            scores = torch.sigmoid(self.gate_ffn(sa))  # N x bs*depth x 1
            scores = scores.permute(1, 0, 2).view(-1, bs, x.shape[0], 1)   # depth x B x N x 1

            # takes only the CLS
            scores = scores[:, :, 0, :]  # depth x B x 1
            orig = orig[:, :, 0, :]  # depth x B x dim
            scores = scores.permute(1, 2, 0)  # B x 1 x depth
            orig = orig.permute(1, 0, 2)  # B x depth x dim
            out = torch.matmul(scores, orig)  # B x 1 x dim
            out = out.squeeze(1)    # B x dim

        if self.proj is not None:
            out = self.proj(out)
        return out
