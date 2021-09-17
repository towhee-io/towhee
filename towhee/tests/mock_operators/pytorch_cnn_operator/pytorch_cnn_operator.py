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

import torch

from towhee.operator import Operator


class PyTorchCNNOperator(Operator):
    """
    PyTorch model operator base
    """
    def __init__(self, model, img_tensor) -> None:
        super().__init__()
        self.model = model
        self.img_tensor = img_tensor

    def __call__(self) -> torch.Tensor:
        super().__call__()
        outputs = self.model(self.img_tensor)
        return outputs

    def train(self):
        """
        For training model
        """
        pass
