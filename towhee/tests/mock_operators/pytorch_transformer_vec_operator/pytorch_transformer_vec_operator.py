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

from typing import NamedTuple
import torch
import numpy

from towhee.operator import Operator


class PytorchTransformerVecOperator(Operator):
    """
    Pytorch transformer vector operator, the output is embedding instead of prediction
        Args:
            model(nn.Module):
                Model object
    """

    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

    def __call__(self, img_tensor: torch.Tensor) -> NamedTuple('Outputs', [('embedding', numpy.ndarray)]):
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(img_tensor).squeeze(0)
        Outputs = NamedTuple('Outputs', [('embedding', numpy.ndarray)])
        return Outputs(outputs)
