# Copyright 2022 Zilliz. All rights reserved.
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
import matplotlib

from pathlib import Path

from towhee.models.visualization.transformer_visualization import generate_attention
from torch import nn

cur_dir = Path(__file__).parent
matplotlib.use("agg")


class MockVit(nn.Module):
    """
    mock vit model
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=16, stride=16, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x.flatten()

    def relprop(self, rel, **kwargs):
        print(kwargs)
        return rel


class TestAttention(unittest.TestCase):
    """
    test attention visualization
    """

    def setUp(self) -> None:
        self.model = MockVit()

    def test_attention(self):
        res = generate_attention(self.model, torch.randn(3, 224, 224), class_index=0)
        self.assertEqual(res.shape[0], 224)


if __name__ == "__main__":
    unittest.main()
