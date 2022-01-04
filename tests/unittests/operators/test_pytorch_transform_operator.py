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
from PIL import Image
from torchvision import transforms
import os

from tests.unittests.mock_operators import PYTORCH_TRANSFORM_OPERATOR_PATH, load_local_operator
from tests.unittests import VIT_DIR


class PytorchTransformTest(unittest.TestCase):
    test_dir = VIT_DIR
    img = Image.open(os.path.join(test_dir, 'img.jpg'))
    tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img1 = tfms(img).unsqueeze(0)

    def test_transform_operator(self):
        trans = load_local_operator('pytorch_transform_operator', PYTORCH_TRANSFORM_OPERATOR_PATH)
        op = trans.PytorchTransformOperator(256)
        outputs = op(self.img)
        c = (self.img1.numpy() == outputs.img_transformed.numpy())
        self.assertEqual(c.all(), True)


if __name__ == '__main__':
    unittest.main()
