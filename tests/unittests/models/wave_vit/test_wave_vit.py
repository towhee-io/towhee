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
from towhee.models.wave_vit.wave_vit import create_model


class TestWaveVit(unittest.TestCase):
    """
    Test WaveVit model
    """
    query_image = torch.randn(1, 3, 224, 224)

    def test_forward_feature(self):
        model = create_model(model_name="wave_vit_s",
                             pretrained=False,
                             token_label=False)

        out = model(self.query_image)
        self.assertTrue(out.shape, (1, 1000))

    def test_forward(self):
        model = create_model(model_name="wave_vit_s",
                             pretrained=False,
                             token_label=True)
        out = model(self.query_image)
        self.assertTrue(out.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
