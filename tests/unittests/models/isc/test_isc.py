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
from towhee.models import isc


class TestIsc(unittest.TestCase):
    """
    Test ISC model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.rand(1, 3, 4, 4).to(device)

    def test_isc(self):
        model = isc.create_model(pretrained=False, device=self.device,
                                 timm_backbone='resnet50', fc_dim=5, p=1.0, eval_p=1.0)
        outs = model(self.data)
        self.assertTrue(outs.shape == (1, 5))


if __name__ == '__main__':
    unittest.main()
