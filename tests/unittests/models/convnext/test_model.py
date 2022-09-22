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
from towhee.models.convnext import create_model


class TestModel(unittest.TestCase):
    """
    Test ConvNeXt model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_model(self):
        data = torch.rand(1, 3, 32, 32).to(self.device)
        model = create_model(
            device=self.device,
            in_chans=3,
            num_classes=5,
            depths=(2, 3, 4, 5),
            dims=(3, 6, 9, 12),
        )
        outs = model(data)
        self.assertTrue(outs.shape == (1, 5))

    def test_model_names(self):
        try:
            create_model(model_name='wrong name')
        except ValueError:
            pass

        tiny_1k = create_model(model_name='convnext_tiny_1k')
        self.assertTrue(tiny_1k.depths == (3, 3, 9, 3))
        self.assertTrue(tiny_1k.dims == (96, 192, 384, 768))
        self.assertTrue(tiny_1k.num_classes == 1000)

        # tiny_22k = create_model(model_name='convnext_tiny_22k')
        # self.assertTrue(tiny_22k.depths == (3, 3, 9, 3))
        # self.assertTrue(tiny_22k.dims == (96, 192, 384, 768))
        # self.assertTrue(tiny_22k.num_classes == 21841)

        small_1k = create_model(model_name='convnext_small_1k')
        self.assertTrue(small_1k.depths == (3, 3, 27, 3))
        self.assertTrue(small_1k.dims == (96, 192, 384, 768))
        self.assertTrue(small_1k.num_classes == 1000)

        # small_22k = create_model(model_name='convnext_small_22k')
        # self.assertTrue(small_22k.depths == (3, 3, 27, 3))
        # self.assertTrue(small_22k.dims == (96, 192, 384, 768))
        # self.assertTrue(small_22k.num_classes == 21841)

        base_1k = create_model(model_name='convnext_base_1k')
        self.assertTrue(base_1k.depths == (3, 3, 27, 3))
        self.assertTrue(base_1k.dims == (128, 256, 512, 1024))
        self.assertTrue(base_1k.num_classes == 1000)

        # base_22k = create_model(model_name='convnext_base_22k')
        # self.assertTrue(base_22k.depths == (3, 3, 27, 3))
        # self.assertTrue(base_22k.dims == (128, 256, 512, 1024))
        # self.assertTrue(base_22k.num_classes == 21841)

        large_1k = create_model(model_name='convnext_large_1k')
        self.assertTrue(large_1k.depths == (3, 3, 27, 3))
        self.assertTrue(large_1k.dims == (192, 384, 768, 1536))
        self.assertTrue(large_1k.num_classes == 1000)

        # large_22k = create_model(model_name='convnext_large_22k')
        # self.assertTrue(large_22k.depths == (3, 3, 27, 3))
        # self.assertTrue(large_22k.dims == (192, 384, 768, 1536))
        # self.assertTrue(large_22k.num_classes == 21841)

        xlarge_22k = create_model(model_name='convnext_xlarge_22k')
        self.assertTrue(xlarge_22k.depths == (3, 3, 27, 3))
        self.assertTrue(xlarge_22k.dims == (256, 512, 1024, 2048))
        self.assertTrue(xlarge_22k.num_classes == 21841)


if __name__ == '__main__':
    unittest.main()
