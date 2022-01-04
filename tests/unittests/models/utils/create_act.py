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
import unittest

from towhee.models.utils.create_act import get_act_layer

class TestCreateAct(unittest.TestCase):
    """
    Test activation layer factory.
    """

    def test_createact(self):
        act_fn = get_act_layer('silu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('swish')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('mish')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('relu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('relu6')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('leaky_relu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('prelu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('celu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('selu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('gelu')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('sigmoid')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('tanh')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('hard_sigmoid')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('hard_swish')
        self.assertTrue(act_fn is not None)

        act_fn = get_act_layer('hard_mish')
        self.assertTrue(act_fn is not None)

