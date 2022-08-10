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

from towhee.models.repmlp import RepMLPNetUnit, RepMLPNet


class TestRepMLP(unittest.TestCase):
    """
    Test RepMLP model
    """
    def test_repmlp_unit(self):
        data = torch.rand(1, 4, 6, 6)
        model = RepMLPNetUnit(channels=4, h=3, w=3, reparam_conv_k=(1, 1), globalperceptron_reduce=4)
        outs = model(data)
        # print(out.shape)
        self.assertTrue(outs.shape == (1, 4, 6, 6))

    def test_repmlp(self):
        data = torch.rand(1, 4, 16, 16)
        model = RepMLPNet(
            in_channels=4, num_class=10,
            patch_size=(4, 4), num_blocks=(2, 4), channels=(2, 4),
            hs=(4, 2), ws=(4, 2),
            sharesets_nums=(2, 4), reparam_conv_k=(1,), globalperceptron_reduce=2,
            use_checkpoint=True, deploy=True)
        self.assertTrue(model.use_checkpoint)

        outs = model(data)
        # print(outs.shape)
        self.assertTrue(outs.shape == (1, 10))

    def test_locality_injection(self):
        data = torch.rand(1, 4, 16, 16)
        model = RepMLPNet(
            in_channels=4, num_class=10,
            patch_size=(4, 4), num_blocks=(2, 4), channels=(2, 4),
            hs=(4, 2), ws=(4, 2),
            sharesets_nums=(2, 4), reparam_conv_k=(1,), globalperceptron_reduce=2,
            use_checkpoint=False, deploy=False)
        model.locality_injection()
        outs = model(data)
        # print(outs.shape)
        self.assertTrue(outs.shape == (1, 10))


if __name__ == '__main__':
    unittest.main()
