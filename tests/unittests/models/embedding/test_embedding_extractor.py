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
import torch
import os
from torchvision.models import resnet34
from torchvision import transforms
from PIL import Image

from towhee.models.embedding.embedding_extractor import EmbeddingExtractor


class EmbeddingExtractorTest(unittest.TestCase):
    def test_embed_extract(self):
        device = torch.device('cpu')
        res = resnet34(pretrained=True)
        res = res.to(device)
        emb = EmbeddingExtractor(res)

        layer_name = 'conv1'
        emb.register(layer_name)
        img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cat.jpg')
        image = Image.open(img)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = transform(image).unsqueeze(dim=0).to(device)
        out = res(img)
        self.assertTrue(emb.emb_out.embeddings[0].shape == torch.Size([1, 1000]))
