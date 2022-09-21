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
from towhee.models.convnext import ConvNeXt


class TestModel(unittest.TestCase):
    """
    Test ConvNeXt model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_model(self):
        data = torch.rand(1, 3, 32, 32).to(self.device)
        model = ConvNeXt(
            in_chans=3,
            num_classes=5,
            depths=(2, 3, 4, 5),
            dims=(3, 6, 9, 12),
        ).to(self.device)
        outs = model(data)
        self.assertTrue(outs.shape == (1, 5))


if __name__ == '__main__':
    unittest.main()
