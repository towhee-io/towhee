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


import unittest
from pathlib import Path
from PIL import Image
import torchvision
import torch
from torch import nn
import numpy as np
from typing import NamedTuple


from towhee.operator.base import PyTorchNNOperator
from towhee.tests.mock_operators.pytorch_transform_operator.pytorch_transform_operator import PytorchTransformOperator


class ResnetOp(PyTorchNNOperator):
    """
    Resnet operator.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def __call__(self, img_tensor: torch.Tensor) -> NamedTuple('outputs', [('cnn', np.ndarray)]):
        outputs = NamedTuple('Outputs', [('cnn', np.ndarray)])
        return outputs(self.model(img_tensor).detach().numpy())


class TestPytorchNNOperator(unittest.TestCase):
    """
    Test PytorchNNOperator
    """

    def test_pytochnn_operator(self):
        cache_path = Path(__file__).parent.parent.resolve()
        test_image = cache_path.joinpath('data/dataset/kaggle_dataset_small/train/001cdf01b096e06d78e9e5112d419397.jpg')
        img_pil = Image.open(test_image)
        op = PytorchTransformOperator(256)
        img_tensor = op(img_pil).img_transformed
        model_name = 'resnet50'
        model_func = getattr(torchvision.models, model_name)
        model = model_func(pretrained=True)
        op = ResnetOp(model)
        self.assertEqual((1, 1000), op(img_tensor)[0].shape)


if __name__ == '__main__':
    unittest.main()
