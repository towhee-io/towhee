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

from towhee.models.repmlp import GlobalPerceptron, RepMLPBlock


class TestBlocks(unittest.TestCase):
    """
    Test RepMLP blocks
    """
    def test_global_perception(self):
        data = torch.rand(3, 1, 1)
        layer = GlobalPerceptron(input_channels=3, internal_neurons=4)
        out = layer(data)
        # print(out.shape)
        self.assertTrue(out.shape == (1, 3, 1, 1))

    def test_repmlp_block(self):
        data = torch.rand(1, 4, 6, 6)
        model = RepMLPBlock(in_channels=4, out_channels=4, h=3, w=3, reparam_conv_k=(1, 1))
        out = model(data)
        # print(out.shape)
        self.assertTrue(out.shape == (1, 4, 6, 6))

        parts1 = model.partition(data, 2, 2)
        # print(parts1.shape)
        self.assertTrue(parts1.shape == (1, 2, 2, 4, 3, 3))
        parts2 = model.partition_affine(data, 2, 2)
        # print(parts2.shape)
        self.assertTrue(parts2.shape == (4, 2, 2, 1, 3, 3))

        # Test local_inject & get_equivalent_fc3 & _convert_conv_to_f
        model.local_inject()
        # print(type(model.fc3_bn), model.fc3.weight.shape, model.fc3.bias.shape)
        self.assertTrue(isinstance(model.fc3_bn, torch.nn.Identity))
        self.assertTrue(model.fc3.weight.shape == (9, 9, 1, 1))
        self.assertTrue(model.fc3.bias.shape == (9,))


if __name__ == '__main__':
    unittest.main()
