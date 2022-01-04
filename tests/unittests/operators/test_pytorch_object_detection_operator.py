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
import torch
import torchvision

from pathlib import Path

from PIL import Image

from tests.unittests.mock_operators import PYTORCH_OBJECT_DETECTION_OPERATOR_PATH, load_local_operator

cache_path = Path(__file__).parent.parent.resolve()
test_image = cache_path.joinpath('data/dataset/kaggle_dataset_small/train/1cef8fb87d5ca2b8a833b5f9549634aa.jpg')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TestOperator(unittest.TestCase):
    """
    Simple operator test
    """
    def setUp(self) -> None:
        img_pil = Image.open(test_image).convert('RGB')
        self.img_tensor = torchvision.transforms.ToTensor()(img_pil)
        self.images = [self.img_tensor.to(device)]
        self.model_name = 'fasterrcnn_resnet50_fpn'

    def test_func_operator(self):
        test_op = load_local_operator(
            'pytorch_object_detection_operator', PYTORCH_OBJECT_DETECTION_OPERATOR_PATH)
        op = test_op.PyTorchObjectDetectionOperator(self.model_name)
        output = op(self.images)
        self.assertEqual(4, output[0].shape[1])


if __name__ == '__main__':
    unittest.main()
