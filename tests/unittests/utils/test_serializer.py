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
import json

import numpy as np
from towhee.runtime.data_queue import Empty
from towhee.utils.serializer import to_triton_data, from_triton_data

class TestSerializer(unittest.TestCase):
    """
    Unit test for serializer.
    """
    def test_np_fail(self):
        with self.assertRaises(TypeError):
            arr = np.arange(4)
            json.dumps(arr)

    def test_empty_fail(self):
        with self.assertRaises(TypeError):
            json.dumps(Empty())

    def test_np(self):
        arr = np.arange(4)
        arr_json = to_triton_data(arr)
        arr_deserialized = from_triton_data(arr_json)
        self.assertTrue((arr==arr_deserialized).all())

    def test_empty(self):
        empty_json = to_triton_data(Empty())
        empty_deserialized = from_triton_data(empty_json)
        self.assertTrue(empty_deserialized is Empty())
