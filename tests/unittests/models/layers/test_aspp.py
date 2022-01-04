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

from towhee.models.layers.aspp import ASPP


class TestOperator(unittest.TestCase):
    """
    A CNN model which adds spp layer so that we can input multi-size tensor
    """

    def setUp(self):
        self.input = torch.rand((2, 512, 13, 13))

    def test_spp(self):
        aspp_module = ASPP()
        self.assertEqual(256, aspp_module(self.input).shape[1])
