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
from towhee.models.hornet import create_model


class TestModel(unittest.TestCase):
    """
    Test HorNet model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_model(self):
        data = torch.rand(1, 3, 32, 32).to(self.device)

        model = create_model(
            num_classes=10, depths=(2, 2, 3, 3), base_dim=16, drop_path_rate=0.1, uniform_init=True
        ).to(self.device)
        outs = model(data)
        self.assertTrue(outs.shape == (1, 10))

    def test_model_names(self):
        try:
            create_model(model_name='wrong name')
        except ValueError:
            pass

        hornet_tiny = create_model(model_name='hornet_tiny_7x7', device=self.device)
        self.assertTrue(hornet_tiny.base_dim == 64)
        self.assertTrue(hornet_tiny.num_classes == 1000)

        hornet_small = create_model(model_name='hornet_small_gf', device=self.device)
        self.assertTrue(hornet_small.base_dim == 96)
        self.assertTrue(hornet_small.num_classes == 1000)

        hornet_base = create_model(model_name='hornet_base_7x7', device=self.device)
        self.assertTrue(hornet_base.base_dim == 128)
        self.assertTrue(hornet_base.num_classes == 1000)

        hornet_large = create_model(model_name='hornet_large_gf_img384_22k', device=self.device)
        self.assertTrue(hornet_large.base_dim == 192)
        self.assertTrue(hornet_large.num_classes == 21841)


if __name__ == '__main__':
    unittest.main()
