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
import torch
import torchvision

from towhee.models.layers.sam import SAM


class TestOperator(unittest.TestCase):
    """
    A test for SAM.
    """

    def setUp(self):
        self.model = torchvision.models.resnet50(pretrained=True)
        self.base_optimizer = torch.optim.SGD

    def test_sam(self):
        optimizer = SAM(self.model.parameters(), self.base_optimizer, lr=0.1, momentum=0.9)
        self.assertEqual(8, len(optimizer.base_optimizer.param_groups[0]))
