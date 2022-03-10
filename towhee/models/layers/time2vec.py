# Original paper:
#
#     "Time2Vec: Learning a Vector Representation of Time" https://arxiv.org/abs/1907.05321
#
# Implemented with Pytorch by Copyright 2022 Zilliz. All rights reserved.
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


class Time2Vec(nn.Module):
    """
    Time2Vec implementaion in Pytorch.

    Args:
        seq_len (`int`):
            the length of input sequence
        activation (`str`):
            activation functions used for periodic time embedding (only support "sin" or "cos")

    Return:
        embedding with the same shape as input

    Example:
        >>> from towhee.models.layers.time2vec import Time2Vec
        >>> import torch
        >>>
        >>> x = torch.randn(3, 64)
        >>> model = Time2Vec(seq_len=64, activation="sin")
        >>> model(x).shape
        torch.Size([3, 64])
    """
    def __init__(self, seq_len: int, activation: str, **kwargs):
        super().__init__(**kwargs)
        self.w0 = nn.parameter.Parameter(torch.randn(seq_len, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(seq_len, 1))
        self.b = nn.parameter.Parameter(torch.randn(1))

        if activation == 'sin':
            self.f = torch.sin
        elif activation == 'cos':
            self.f = torch.cos
        else:
            raise ValueError(f'Activation {activation} is not supported yet.')

        self.fc = nn.Linear(2, seq_len)

    def forward(self, x):
        periodic = self.f(torch.matmul(x, self.w) + self.b)  # Periodic time embedding
        linear = torch.matmul(x, self.w0) + self.b0  # Linear time embedding
        out = torch.cat([periodic, linear], -1)  # Concat time embeddings
        out = self.fc(out)  # Convert embedding dimension back to sequence length
        return out
