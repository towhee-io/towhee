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

import towhee
from towhee.dc2 import ops, register, accelerate


@accelerate
class Model:
    """
    Model
    """
    def __init__(self, x):
        self._x = x

    def __call__(self, num):
        return self._x + num


@register
class MyAccOp:
    """
    Test acc op
    """

    def __init__(self, x):
        self._mode = Model(x)

    def __call__(self, num):
        return self._mode(num)


class TestAccelerator(unittest.TestCase):
    """
    Test accelerator
    """
    def test_normal(self):
        p = (towhee.pipe.input('a')
             .map('a', 'c', ops.MyAccOp(10))
             .output('c'))
        self.assertEqual(p(10).get()[0], 20)

        p = (towhee.pipe.input('a')
             .map('a', 'c', ops.MyAccOp(10),
                  config={'acc_info': {'type': 'mock', 'params': None}})
             .output('c'))

        # the mock accelerator will do nothing
        self.assertEqual(p(10).get()[0], 10)
