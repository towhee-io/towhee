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
from towhee.models.movinet.movinet import MoViNet, create_model

class MovinetTest(unittest.TestCase):
    def test_Movinet_A0(self):
        """
        Test MovinetA0 model.
        """
        model = create_model(model_name='movineta0',causal=False)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A0_stream(self):
        """
        Test MovinetA0 stream model.
        """
        model = create_model(model_name='movineta0',causal=True)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A1(self):
        """
        Test MovinetA1 model.
        """
        model = create_model(model_name='movineta1',causal=False)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A1_stream(self):
        """
        Test MovinetA1 stream model.
        """
        model = create_model(model_name='movineta1',causal=True)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A2(self):
        """
        Test MovinetA2 model.
        """
        model = create_model(model_name='movineta2',causal=False)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A2_stream(self):
        """
        Test MovinetA2 stream model.
        """
        model = create_model(model_name='movineta2',causal=True)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A3(self):
        """
        Test MovinetA3 model.
        """
        model = create_model(model_name='movineta3',causal=False)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A3_stream(self):
        """
        Test MovinetA3 model.
        """
        model = create_model(model_name='movineta3',causal=True)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A4(self):
        """
        Test MovinetA4 model.
        """
        model = create_model(model_name='movineta4',causal=False)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A4_stream(self):
        """
        Test MovinetA4 model.
        """
        model = create_model(model_name='movineta4',causal=True)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A5(self):
        """
        Test MovinetA5 model.
        """
        model = create_model(model_name='movineta5',causal=False)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A5_stream(self):
        """
        Test MovinetA5 model.
        """
        model = create_model(model_name='movineta5',causal=True)
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A4_(self):
        """
        Test MovinetA4 2+1d model.
        """
        config = _C.MODEL.MoViNetA4
        model = MoViNet(cfg = config, causal = True, conv_type = '2plus1d')
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))

    def test_Movinet_A5_(self):
        """
        Test MovinetA5 2+1d model.
        """
        config = _C.MODEL.MoViNetA5
        model = MoViNet(cfg=config, causal = False, conv_type = '2plus1d')
        x=torch.randn(1,3,1,64,64)
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,600]))
