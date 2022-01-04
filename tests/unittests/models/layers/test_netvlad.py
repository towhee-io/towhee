# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team and 2021 Zilliz.
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
import torchvision

from pathlib import Path
from PIL import Image
from towhee.models.layers.netvlad import NetVLAD, EmbedNet
from torchvision.models import resnet18
from torch import nn


cache_path = Path(__file__).parent.parent.parent.resolve()
test_image = cache_path.joinpath('data/dataset/kaggle_dataset_small/train/001cdf01b096e06d78e9e5112d419397.jpg')


class NetvladTest(unittest.TestCase):
    def setUp(self) -> None:
        # Discard layers at the end of base network
        encoder = resnet18(pretrained=True)
        base_model = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        self.dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)
        self.num_clusters = 32

        # Define model for embedding, K clusters determined by num_clusters
        net_vlad = NetVLAD(num_clusters=self.num_clusters, dim=self.dim, alpha=1.0)
        self.model = EmbedNet(base_model, net_vlad)
        img_pil = Image.open(test_image)
        self.img_tensor = torchvision.transforms.ToTensor()(img_pil).unsqueeze(0)

    def test_netvlad(self):
        # x = torch.rand(1, 3, 256, 436)
        output = self.model(self.img_tensor)
        self.assertEqual(self.num_clusters * self.dim, output.shape[1])


if __name__ == '__main__':
    unittest.main(verbosity=1)
