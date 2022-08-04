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
import unittest
import torch
from towhee.models.wave_vit.wave_vit_utils import IDWT2D, DWT2D


class TestWaveVitUtils(unittest.TestCase):
    """
    Test svt model
    """

    def test_IDWT2D(self):
        xx = torch.randn(1, 3, 24, 24)
        dwt = DWT2D(wave="haar")
        dwt.float()
        out = dwt(xx)
        self.assertTrue(out.shape, (1, 12, 12, 12))

    def test_DWT2D(self):
        xx = torch.randn(1, 12, 12, 12)
        i_dwt = IDWT2D(wave="haar")
        i_dwt.float()
        out = i_dwt(xx)
        self.assertTrue(out.shape, (1, 3, 24, 24))


if __name__ == "__main__":
    unittest.main()




