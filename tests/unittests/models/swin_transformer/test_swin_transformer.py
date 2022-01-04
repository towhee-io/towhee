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

from towhee.models.swin_transformer.configs import build_configs
from towhee.models.swin_transformer.model import SwinTransformer
from towhee.models.utils.pretrained_utils import load_pretrained_weights

from tests.unittests import MODEL_RESOURCE


class SwinTransformerTest(unittest.TestCase):
    name = 'swin_small_patch4_window7_224'
    resource_dir = MODEL_RESOURCE
    arch, model_cfg = build_configs(name)
    model = SwinTransformer(**arch)

    load_pretrained_weights(model, name, model_cfg)
    img = Image.open(os.path.join(resource_dir, 'img.jpg'))
    tfms = transforms.Compose([transforms.Resize([256, 256]),
                               transforms.CenterCrop([224, 224]),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    img = tfms(img).unsqueeze(0)
    with open(os.path.join(resource_dir, 'labels_map.txt'), encoding='utf-8') as handler:
        labels_map = json.load(handler)
    _labels_map = []

    for i in range(1000):
        _labels_map.append(labels_map[str(i)])
    labels_map = _labels_map
    model.eval()
    out = model(img)
    scores = torch.nn.functional.softmax(out, 1)
    with torch.no_grad():
        outputs = model(img).squeeze(0)

    def test_swin_transformer(self):
        for idx in torch.topk(self.outputs, k=1).indices.tolist():
            #prob = torch.softmax(self.outputs, -1)[idx].item()
            #label = self.labels_map[idx]
            #p = prob * 100
            self.assertEqual('giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', self.labels_map[idx])


if __name__ == '__main__':
    unittest.main()
