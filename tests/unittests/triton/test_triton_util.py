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

from towhee.types import Image, AudioFrame, VideoFrame
from towhee.serve.triton.python_backend_wrapper import MockTritonPythonBackendTensor
from towhee.serve.triton.triton_util import ToTowheeData


class TestToTowheeData(unittest.TestCase):
    def test_to_towhee_data_image(self):
        data = {
            'INPUT0': MockTritonPythonBackendTensor(np.array([10])),
            'INPUT1': MockTritonPythonBackendTensor(np.random.randn(224, 224, 3)),
            'INPUT2': MockTritonPythonBackendTensor(np.array(['RGB'.encode('utf-8')], dtype=np.object_)),
        }

        schema = [(int, (1,)), (Image, (-1, -1, 3))]
        data = ToTowheeData(data, schema).get_towhee_data()
        self.assertEqual(len(data), 2)
        self.assertTrue(isinstance(data[0], int))
        self.assertEqual(data[0], 10)
        
        self.assertTrue(isinstance(data[1], Image))
        self.assertEqual(data[1].mode, 'RGB')
        self.assertEqual(data[1].shape, (224, 224, 3))


    def test_to_towhee_data_audioframe(self):
        data = {
            'INPUT0': MockTritonPythonBackendTensor(np.random.randn(224, 224, 3)),
            'INPUT1': MockTritonPythonBackendTensor(np.array([1000])),
            'INPUT2': MockTritonPythonBackendTensor(np.array([2000])),
            'INPUT3': MockTritonPythonBackendTensor(np.array(['mono'.encode('utf-8')], dtype=np.object_)),
            'INPUT4': MockTritonPythonBackendTensor(np.array([10.5])),
        }

        schema = [(AudioFrame, (-1, -1, 3)), (float, (1,))]
        data = ToTowheeData(data, schema).get_towhee_data()
        self.assertEqual(len(data), 2)

        self.assertTrue(isinstance(data[0], AudioFrame))
        self.assertEqual(data[0].layout, 'mono')
        self.assertEqual(data[0].shape, (224, 224, 3))
        self.assertEqual(data[0].timestamp, 2000)
        self.assertEqual(data[0].sample_rate, 1000)

        self.assertTrue(isinstance(data[1], float))
        self.assertEqual(data[1], 10.5)

    def test_to_towhee_data_videoframe(self):
        data = {
            'INPUT0': MockTritonPythonBackendTensor(np.random.randn(224, 224, 3)),
            'INPUT1': MockTritonPythonBackendTensor(np.array(['RGB'.encode('utf-8')], dtype=np.object_)),
            'INPUT2': MockTritonPythonBackendTensor(np.array([1000])),
            'INPUT3': MockTritonPythonBackendTensor(np.array([1])),
            'INPUT4': MockTritonPythonBackendTensor(np.array([1, 2])),
        }

        schema = [(VideoFrame, (-1, -1, 3)), (np.int32, (-1,))]
        data = ToTowheeData(data, schema).get_towhee_data()
        self.assertEqual(len(data), 2)

        self.assertTrue(isinstance(data[0], VideoFrame))
        self.assertEqual(data[0].mode, 'RGB')
        self.assertEqual(data[0].shape, (224, 224, 3))
        self.assertEqual(data[0].timestamp, 1000)
        self.assertEqual(data[0].key_frame, 1)
        self.assertEqual(data[1].tolist(), [1, 2])
