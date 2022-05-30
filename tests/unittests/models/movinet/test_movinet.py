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
from towhee.models.movinet.config import _C
from towhee.models.movinet.movinet import MoViNet

class MovinetTest(unittest.TestCase):
    def test_Movinet_A0(self):
        """
        Test MovinetA0 model.
        """
        config = _C.MODEL.MoViNetA0
        model = MoViNet(cfg=config)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A1(self):
        """
        Test MovinetA1 model.
        """
        config = _C.MODEL.MoViNetA1
        model = MoViNet(cfg=config)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A2(self):
        """
        Test MovinetA2 model.
        """
        config = _C.MODEL.MoViNetA2
        model = MoViNet(cfg=config)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A3(self):
        """
        Test MovinetA3 model.
        """
        config = _C.MODEL.MoViNetA3
        model = MoViNet(cfg=config)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A4(self):
        """
        Test MovinetA4 model.
        """
        config = _C.MODEL.MoViNetA4
        model = MoViNet(cfg=config)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A5(self):
        """
        Test MovinetA5 model.
        """
        config = _C.MODEL.MoViNetA5
        model = MoViNet(cfg=config)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))
