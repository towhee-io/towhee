# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import json
from PIL import Image
import torch
from torchvision import transforms

from towhee.models.vit.vit import VitModel
from tests.unittests import VIT_DIR


class VisionTransformerTest(unittest.TestCase):
    name = 'vit_base_patch16_224'
    test_dir = VIT_DIR
    weights_path = None
    model = VitModel(name, pretrained=True)
    img = Image.open(os.path.join(test_dir, 'img.jpg'))
    tfms = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
    img = tfms(img).unsqueeze(0)
    with open(os.path.join(test_dir, 'labels_map.txt'), encoding='utf-8') as handler:
        labels_map = json.load(handler)
    # labels_map = json.load('labels_map.txt')
    _labels_map = []
    for i in range(1000):
        _labels_map.append(labels_map[str(i)])
    labels_map = _labels_map
    # labels_map = [labels_map[str(i)] for i in range(1000)]
    with torch.no_grad():
        outputs = model(img).squeeze(0)

    def test_pretrained(self):
        for idx in torch.topk(self.outputs, k=1).indices.tolist():
            prob = torch.softmax(self.outputs, dim=0)[idx].item()
            label = self.labels_map[idx]
            p = prob * 100
            print(f'[{idx}] {label:<75} ({p:.2f}%)')
            self.assertEqual('giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', self.labels_map[idx])


if __name__ == '__main__':
    unittest.main()
