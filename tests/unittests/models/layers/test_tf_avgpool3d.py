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
from towhee.models.layers.tf_avgpool3d import TfAvgPool3D

class TfAvgPool3DTest(unittest.TestCase):
    def test_tf_avgpool3d(self):
        """
        Test tf avgpool3d layer.
        """
        x=torch.randn(1,1,1,4,4)
        model = TfAvgPool3D()
        y=model(x)
        self.assertTrue(y.shape == torch.Size([1,1,1,2,2]))
