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

import numpy as np
import torch
import torchvision

from towhee.operator import Operator


class PytorchCnnOperator(Operator):
    """
    PyTorch model operator base
    """
    def __init__(self, model_name) -> None:
        super().__init__()
        model_func = getattr(torchvision.models, model_name)
        self._model = model_func(pretrained=True)
        self._model.eval()

    def __call__(self, img_tensor: torch.Tensor) -> NamedTuple('Outputs', [('cnn', np.ndarray)]):
        Outputs = NamedTuple('Outputs', [('cnn', np.ndarray)])
        return Outputs(self._model(img_tensor).detach().numpy())

    def train(self):
        """
        For training model
        """
        pass
