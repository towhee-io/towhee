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
from towhee.models import svt


class TestSVT(unittest.TestCase):
    """
    Test svt model
    """
    def test_svt(self):
        video = torch.randn(1, 3, 8, 32, 32)
        model = svt.create_model(
            model_name='svt_vitb_k400',
            pretrained=False,
            img_size=32)
        feats = model.forward_features(video)
        output = model(video)
        self.assertTrue(feats.shape == (1, 768))
        self.assertTrue(output.shape == (1, 400))


if __name__ == '__main__':
    unittest.main()
