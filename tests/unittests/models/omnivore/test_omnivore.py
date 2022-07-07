# Copyright 2022 Zilliz. All rights reserved.
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

import torch
import unittest
from towhee.models.omnivore.omnivore import omnivore_swins, omnivore_swint, omnivore_swinb_imagenet21k, \
    omnivore_swinl_imagenet21k


class OmnivoreTest(unittest.TestCase):
    def test_omnivore_swins(self):
        pretrained = False
        model = omnivore_swins(pretrained=pretrained)
        x = torch.randn(1, 3, 5, 4, 4)
        y = model(x, "video")
        self.assertTrue(y.shape == torch.Size([1, 400]))

    def test_omnivore_swint(self):
        pretrained = False
        model = omnivore_swint(pretrained=pretrained)
        x = torch.randn(1, 3, 5, 4, 4)
        y = model(x, "video")
        self.assertTrue(y.shape == torch.Size([1, 400]))

    def test_omnivore_swinb_imagenet21k(self):
        pretrained = False
        model = omnivore_swinb_imagenet21k(pretrained=pretrained)
        x = torch.randn(1, 3, 5, 4, 4)
        y = model(x, "video")
        self.assertTrue(y.shape == torch.Size([1, 400]))

    def test_omnivore_swinl_imagenet21k(self):
        pretrained = False
        model = omnivore_swinl_imagenet21k(pretrained=pretrained)
        x = torch.randn(1, 3, 5, 4, 4)
        y = model(x, "video")
        self.assertTrue(y.shape == torch.Size([1, 400]))


if __name__ == "__main__":
    unittest.main()
