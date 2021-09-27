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
from torchvision import transforms

from towhee.operator import Operator


class PytorchTransformOperator(Operator):
    """
    Image transform operator
        Args:
            tfms(transforms.Compose):
                This is used to transform an image.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        # user defined transform
        self.tfms = transforms.Compose([transforms.Resize(size),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])

    def __call__(self, img_tensor: torch.Tensor) -> NamedTuple('Outputs', [('img_transformed', torch.Tensor)]):
        Outputs = NamedTuple('Outputs', [('img_transformed', torch.Tensor)])
        return Outputs(self.tfms(img_tensor).unsqueeze(0))
