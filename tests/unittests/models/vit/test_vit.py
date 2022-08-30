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
import torch
from tests.unittests import VIT_DIR
from torchvision import transforms

from towhee.utils.pil_utils import PILImage as Image
from towhee.models import vit


class VisionTransformerTest(unittest.TestCase):
    test_dir = VIT_DIR
    model = vit.create_model(model_name='vit_base_16x224', pretrained=True)
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
    outputs = model(img).squeeze(0)

    def test_pretrained(self):
        for idx in torch.topk(self.outputs, k=1).indices.tolist():
            prob = torch.softmax(self.outputs, dim=0)[idx].item()
            label = self.labels_map[idx]
            p = prob * 100
            print(f'[{idx}] {label:<75} ({p:.2f}%)')
            self.assertEqual('giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', self.labels_map[idx])

    def test_model(self):
        dummy_img = torch.rand(1, 3, 4, 4)
        model = vit.create_model(num_classes=400, img_size=4, patch_size=2)
        features = model.forward_features(dummy_img)
        self.assertTrue(features.shape, (1, 768))
        out = model(dummy_img)
        self.assertEqual(out.shape, (1, 400))


if __name__ == '__main__':
    unittest.main()
