# Implementation of TransRAC in paper:
#   [TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting]
#   (https://arxiv.org/abs/2204.01018)
#
# Inspired by official code from https://github.com/SvipRepetitionCounting/TransRAC
#
# Modifications & additions by Copyright 2021 Zilliz. All rights reserved.
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

from torch import nn
from towhee.models.layers.attention import Attention


class DenseMap(nn.Module):
    """
    Predict the density map with DenseNet

    Example:
        >>> import torch
        >>> from towhee.models.transrac import DenseMap
        >>>
        >>> dummy_input = torch.rand(3)
        >>> dense_map = DenseMap(input_dim=3, hidden_dim_1=8, hidden_dim_2=8, out_dim=5)
        >>> out = dense_map(dummy_input)
        >>> print(out.shape)
        torch.Size([5])
    """
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, out_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.Dropout(p=dropout, inplace=False),
            nn.ReLU(True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_dim_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class SimilarityMatrix(nn.Module):
    """
    Build similarity matrix for TransRAC
    """

    def __init__(self, num_heads=4, input_dim=512, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = input_dim
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = Attention(att_dropout=0.)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        query = query.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        key = key.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        value = value.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix, _ = self.attention(query, key, value, attn_mask)

        return matrix
