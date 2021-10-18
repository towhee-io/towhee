# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import unittest
import torch
from torch import nn

from towhee.models.layers.mlp import MLP


class TestMLP(unittest.TestCase):
    """
    test MLP layer
    """
    def test_MLP(self):
        in_features = [10, 20, 30]
        hidden_features = [10, 20, 20]
        out_features = [10, 20, 30]
        act_layers = [nn.GELU, nn.ReLU, nn.Sigmoid]
        drop_rates = [0.0, 0.1, 0.5]
        batch_size = 8
        for in_feat, hidden_feat, out_feat, act_layer, drop_rate in itertools.product(
            in_features, hidden_features, out_features, act_layers, drop_rates
        ):
            mlp_block = MLP(
                in_features=in_feat,
                hidden_features=hidden_feat,
                out_features=out_feat,
                act_layer=act_layer,
                dropout_rate=drop_rate,
            )
            fake_input = torch.rand((batch_size, in_feat))
            output = mlp_block(fake_input)
            self.assertTrue(output.shape, torch.Size([batch_size, out_feat]))
