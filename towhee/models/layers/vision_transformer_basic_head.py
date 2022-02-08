# Copyright 2021  Facebook. All rights reserved.
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
from torch import nn

from towhee.models.layers.sequence_pool import SequencePool


class VisionTransformerBasicHead(nn.Module):
    """
    Vision transformer basic head.
    ::
                                      SequencePool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
    Args:
        in_features ('int'):
            Input channel size of the resnet head.
        out_features ('int'):
            Output channel size of the resnet head.
        seq_pool_type ('str'):
            Pooling type. It supports "cls", "mean " and "none". If set to
            "cls", it assumes the first element in the input is the cls token and
            returns it. If set to "mean", it returns the mean of the entire sequence.
        activation ('callable'):
            A callable that constructs vision transformer head activation layer,
            examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not applying activation).
        dropout_rate ('float'):
            Dropout rate.
    """

    def __init__(
        self,
        *,
        in_features,
        out_features,
        seq_pool_type="cls",
        dropout_rate=0.5,
        activation=None,
    ) -> None:
        super().__init__()
        assert seq_pool_type in ["cls", "mean", "none"]

        if seq_pool_type in ["cls", "mean"]:
            self.seq_pool_model = SequencePool(seq_pool_type)
        elif seq_pool_type == "none":
            self.seq_pool_model = None
        else:
            raise NotImplementedError

        if activation is None:
            self.activation_model = None
        elif activation == nn.Softmax:
            self.activation_model = self.activation(dim=1)
        else:
            self.activation_model = self.activation()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        if self.seq_pool_model is not None:
            x = self.seq_pool_model(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        x = self.proj(x)
        # Performs activation.
        if self.activation_model is not None:
            x = self.activation_model(x)
        return x
