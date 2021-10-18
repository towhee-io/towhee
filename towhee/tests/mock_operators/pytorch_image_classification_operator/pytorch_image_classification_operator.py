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
import json

import numpy
import torch

from typing import NamedTuple
from pathlib import Path

import torchvision

from towhee.operator import Operator

cache_path = Path(__file__).parent.parent.parent.resolve()


def decode_predictions(preds):
    fpath = cache_path.joinpath('data/dataset/imagenet_index_path.json')
    with open(fpath, encoding='ascii') as f:
        class_index = json.load(f)

    results = tuple(class_index[str(preds)])[1]
    return results


class PyTorchImageClassificationOperator(Operator):
    """
    PyTorch model operator base
    """

    def __init__(self, model_name) -> None:
        super().__init__()
        model_func = getattr(torchvision.models, model_name)
        self._model = model_func(pretrained=True)
        self._model.eval()

    def __call__(self, img_tensor: torch.Tensor) -> NamedTuple('Outputs', [('embedding', numpy.ndarray),
                                                                           ('breed', str)]):
        Outputs = NamedTuple('Outputs', [('embedding', numpy.ndarray), ('breed', str)])
        outputs = self._model(img_tensor)
        _, preds = torch.max(outputs, 1)
        results = decode_predictions(int(preds))
        return Outputs(outputs.detach().numpy(), results)

    def train(self):
        """
        For training model
        """
        pass
