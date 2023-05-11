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

import numpy as np

from towhee.utils.serializer import to_triton_data, from_triton_data


class TestNPFormat(unittest.TestCase):
    """
    Test np <---> json
    """
    def test_normal(self):
        arr = np.random.rand(2, 4, 4)
        test_data = ['test', 1, arr, [1, 'a', [1, 2]], [arr] * 2, {'1': arr, 'b': [arr]}]
        s = to_triton_data(test_data)

        ret = from_triton_data(s)

        self.assertTrue((ret[2] == arr).all())
        self.assertTrue((ret[-1]['1'] == arr).all())

    def test_unsupport(self):
        arr = np.random.rand(2, 4, 4)
        arr = arr.astype(np.float128)
        with self.assertRaises(ValueError):
            to_triton_data(arr)
