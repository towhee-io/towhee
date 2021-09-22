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

import torchvision
from PIL import Image

from towhee.tests.mock_operators import PYTORCH_CNN_OPERATOR_PATH, load_local_operator
from torchvision import transforms


cache_path = Path(__file__).parent.parent.resolve()
test_image = cache_path.joinpath('dataset/kaggle_dataset_small/train/001cdf01b096e06d78e9e5112d419397.jpg')

model = torchvision.models.resnet50(pretrained=True)


class TestOperator(unittest.TestCase):
    """
    Simple operator test
    """
    def setUp(self) -> None:
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.model = model
        img_pil = Image.open(test_image)
        self.img_tensor = data_transforms['test'](img_pil)
        self.img_tensor.unsqueeze_(0)

    def test_func_operator(self):
        pytorch_cnn_operator = load_local_operator(
            'pytorch_cnn_operator', PYTORCH_CNN_OPERATOR_PATH)
        op = pytorch_cnn_operator.PyTorchCNNOperator(self.model, self.img_tensor)
        self.assertEqual((1, 1000), op()[0].shape)


if __name__ == '__main__':
    unittest.main()
