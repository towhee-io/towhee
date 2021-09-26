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

from towhee.operator import Operator


class PytorchTransformOperator(Operator):
    """
    Image transform operator
        Args:
            tfms(transforms.Compose):
                This is used to transform an image.
    """

    def __init__(self, tfms) -> None:
        super().__init__()
        self._tfms = tfms

    def __call__(self, img_tensor: torch.Tensor) -> NamedTuple('Outputs', [('img_transformed', torch.Tensor)]):
        Outputs = NamedTuple('Outputs', [('img_transformed', torch.Tensor)])
        return Outputs(self._tfms(img_tensor).unsqueeze(0))
