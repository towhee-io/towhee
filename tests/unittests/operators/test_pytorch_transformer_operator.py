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
import json
import os
from PIL import Image
from torchvision import transforms


from towhee.trainer.models.vit.vit import ViT
from tests.unittests.mock_operators import PYTORCH_TRANSFORMER_OPERATOR_PATH, load_local_operator
from tests.unittests import VIT_DIR


class TransformerOperatorTest(unittest.TestCase):
    name = 'B_16_imagenet1k'
    test_dir = VIT_DIR
    #weights_path = test_dir + 'B_16_imagenet1k.pth'
    weights_path = None
    model = ViT(name, weights_path=weights_path, pretrained=True)
    img = Image.open(os.path.join(test_dir, 'img.jpg'))
    tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
    img = tfms(img).unsqueeze(0)
    with open(os.path.join(test_dir, 'labels_map.txt'), encoding='utf-8') as handler:
        labels_map = json.load(handler)
    _labels_map = []
    for i in range(1000):
        _labels_map.append(labels_map[str(i)])
    labels_map = _labels_map
    args = {'topk': 1, 'labels_map': labels_map}

    def test_transformer_operator(self):
        trans = load_local_operator(
            'pytorch_transformer_operator', PYTORCH_TRANSFORMER_OPERATOR_PATH)
        op = trans.PytorchTransformerOperator(self.model, self.args)
        outputs = op(self.img)
        self.assertEqual(
            'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', outputs.predict)


if __name__ == '__main__':
    unittest.main()
