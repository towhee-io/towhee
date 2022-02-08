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


class SequencePool(nn.Module):
    """
    Sequence pool produces a single embedding from a sequence of embeddings.
    Currently it supports "mean" and "cls".
    """

    def __init__(self, mode: str) -> None:
        """
        Args:
            mode ('str'):
                If set to "cls", it assumes the first element in the input is the cls token and returns it.
                If set to "mean", it returns the mean of the entire sequence.
        """
        super().__init__()
        self.mode = mode
        assert mode in ["cls", "mean"], "Unsupported mode for SequencePool."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "cls":
            x = x[:, 0]
        elif self.mode == "mean":
            x = x.mean(1)
        else:
            raise NotImplementedError
        return x
