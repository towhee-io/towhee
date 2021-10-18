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
import json
from PIL import Image
import torch
from torchvision import transforms
import os

from towhee.models.vit.vit import ViT


class VisionTransformerTest(unittest.TestCase):
    name = 'B_16_imagenet1k'
    pwd = os.getcwd()
    print('pwd is ' + str(pwd))
    test_dir = pwd+'/towhee/tests/models/vit/'
    #weights_path = test_dir + 'B_16_imagenet1k.pth'
    weights_path = None
    model = ViT(name, weights_path=weights_path, pretrained=True)
    img = Image.open(test_dir + 'img.jpg')
    tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
    img = tfms(img).unsqueeze(0)
    with open(test_dir + 'labels_map.txt', encoding='utf-8') as handler:
        labels_map = json.load(handler)
    #labels_map = json.load('labels_map.txt')
    _labels_map = []
    for i in range(1000):
        _labels_map.append(labels_map[str(i)])
    labels_map = _labels_map
    #labels_map = [labels_map[str(i)] for i in range(1000)]
    model.eval()
    with torch.no_grad():
        outputs = model(img).squeeze(0)

    def test_pretrained(self):
        print('-----')
        for idx in torch.topk(self.outputs, k=1).indices.tolist():
            prob = torch.softmax(self.outputs, -1)[idx].item()
            label = self.labels_map[idx]
            p = prob * 100
            print(f'[{idx}] {label:<75} ({p:.2f}%)')
            self.assertEqual('giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', self.labels_map[idx])


if __name__ == '__main__':
    unittest.main()
