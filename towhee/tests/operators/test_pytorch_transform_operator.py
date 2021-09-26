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
from torchvision import models
import os

from towhee.tests.mock_operators import PYTORCH_TRANSFORME_OPERATOR_PATH, load_local_operator


class PytorchTransformTest(unittest.TestCase):
    pwd = os.getcwd()
    print('pwd is ' + str(pwd))
    test_dir = pwd+'/towhee/tests/models/vit/'
    vgg = models.vgg11()
    img = Image.open(test_dir + 'img.jpg')
    tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
    img1 = tfms(img).unsqueeze(0)

    def test_transform_operator(self):
        print('-----')
        trans = load_local_operator('pytorch_transform_operator', PYTORCH_TRANSFORME_OPERATOR_PATH)
        op = trans.PytorchTransformOperator(self.tfms)
        outputs = op(self.img)
        print(type(outputs.img_transformed))
        c = (self.img1.numpy() == outputs.img_transformed.numpy())
        self.assertEqual(c.all(), True)


if __name__ == '__main__':
    unittest.main()
