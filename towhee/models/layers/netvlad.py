# Copyright lyakaap. All rights reserved.
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
# This code is modified by Zilliz.

import torch
import torch.nn.functional as F

from torch import nn


class NetVLAD(nn.Module):
    """
    NetVLAD layer implementation.

    Args:
            num_clusters (`int`): The number of clusters
            dim (`int`): Dimension of descriptors
            alpha (`float`): Parameter of initialization. Larger value is harder assignment.
            normalize_input (`bool`): If true, descriptor-wise L2 normalization is applied to input.
    """

    def __init__(self, num_clusters: int = 64, dim: int = 128, alpha: float = 100.0,
                 normalize_input: bool = True):
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x: torch.Tensor):
        num_sample, in_dim = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(num_sample, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(num_sample, in_dim, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class EmbedNet(nn.Module):
    """
    Embed a base model and the net vlad to a new network.

    Args:
            base_model (`int`): The base model that extracts image features
            net_vlad (`int`): NetVLAD model to extract global features
    """
    def __init__(self, base_model: torch.nn.Sequential, net_vlad: NetVLAD):
        super().__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x
