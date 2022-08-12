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
from towhee.serve.triton.triton_util import ToTowheeData, ToTritonData


class TestToTowheeData(unittest.TestCase):
    '''
    Test ToTowheeData
    '''

    def test_to_towhee_data_image(self):
        data = {
            'INPUT0': MockTritonPythonBackendTensor('INPUT0', np.array([10])),
            'INPUT1': MockTritonPythonBackendTensor('INPUT1', np.random.randn(224, 224, 3)),
            'INPUT2': MockTritonPythonBackendTensor('INPUT2', np.array(['RGB'.encode('utf-8')], dtype=np.object_)),
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
            'INPUT0': MockTritonPythonBackendTensor('INPUT0', np.random.randn(224, 224, 3)),
            'INPUT1': MockTritonPythonBackendTensor('INPUT1', np.array([1000])),
            'INPUT2': MockTritonPythonBackendTensor('INPUT2', np.array([2000])),
            'INPUT3': MockTritonPythonBackendTensor('INPUT3', np.array(['mono'.encode('utf-8')], dtype=np.object_)),
            'INPUT4': MockTritonPythonBackendTensor('INPUT4', np.array([10.5])),
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
            'INPUT0': MockTritonPythonBackendTensor('INPUT0', np.random.randn(224, 224, 3)),
            'INPUT1': MockTritonPythonBackendTensor('INPUT1', np.array(['RGB'.encode('utf-8')], dtype=np.object_)),
            'INPUT2': MockTritonPythonBackendTensor('INPUT2', np.array([1000])),
            'INPUT3': MockTritonPythonBackendTensor('INPUT3', np.array([1])),
            'INPUT4': MockTritonPythonBackendTensor('INPUT4', np.array([1, 2])),
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

    def test_to_towhee_data_error(self):
        data = {
            'INPUT0': MockTritonPythonBackendTensor('INPUT0', np.array([1]))
        }
        schema = [(MockTritonPythonBackendTensor, (1, ))]
        data = ToTowheeData(data, schema).get_towhee_data()
        self.assertTrue(data is None)


class TestToTritonData(unittest.TestCase):
    '''
    Test ToTritonData.
    '''

    def test_to_triton_data_image(self):
        towhee_data = [Image(np.random.randn(224, 224, 3), 'RGB'), 100]
        triton_datas = ToTritonData(towhee_data).get_triton_tensor('INPUT')
        self.assertEqual(len(triton_datas), 3)
        self.assertEqual(triton_datas[0].name(), 'INPUT0')
        self.assertEqual(triton_datas[0].as_numpy().shape, (224, 224, 3))
        self.assertEqual(triton_datas[1].name(), 'INPUT1')
        self.assertEqual(triton_datas[1].as_numpy()[0].decode('utf-8'), 'RGB')
        self.assertEqual(triton_datas[2].name(), 'INPUT2')
        self.assertEqual(triton_datas[2].as_numpy()[0], 100)

    def test_to_triton_data_videoframe(self):
        towhee_data = [32.5, 'BGR', VideoFrame(np.random.randn(224, 224, 3), 'RGB', 10000, 1)]
        triton_datas = ToTritonData(towhee_data).get_triton_tensor('INPUT')
        self.assertEqual(len(triton_datas), 6)
        self.assertEqual(triton_datas[0].name(), 'INPUT0')
        self.assertEqual(triton_datas[0].as_numpy()[0], 32.5)
        self.assertEqual(triton_datas[1].name(), 'INPUT1')
        self.assertEqual(triton_datas[1].as_numpy()[0].decode('utf-8'), 'BGR')
        self.assertEqual(triton_datas[2].name(), 'INPUT2')
        self.assertEqual(triton_datas[2].as_numpy().shape, (224, 224, 3))
        self.assertEqual(triton_datas[3].name(), 'INPUT3')
        self.assertEqual(triton_datas[3].as_numpy()[0].decode('utf-8'), 'RGB')
        self.assertEqual(triton_datas[4].name(), 'INPUT4')
        self.assertEqual(triton_datas[4].as_numpy()[0], 10000)
        self.assertEqual(triton_datas[5].name(), 'INPUT5')
        self.assertEqual(triton_datas[5].as_numpy()[0], 1)

    def test_to_triton_data_audioframe(self):
        towhee_data = [np.random.randn(3, 4), AudioFrame(np.random.randn(224, 224, 3), 1000, 2000, 'mono')]
        triton_datas = ToTritonData(towhee_data).get_triton_tensor('INPUT')
        self.assertEqual(len(triton_datas), 5)
        self.assertEqual(triton_datas[0].name(), 'INPUT0')
        self.assertEqual(triton_datas[0].as_numpy().shape, (3, 4))
        self.assertEqual(triton_datas[1].name(), 'INPUT1')
        self.assertEqual(triton_datas[1].as_numpy().shape, (224, 224, 3))
        self.assertEqual(triton_datas[2].name(), 'INPUT2')
        self.assertEqual(triton_datas[2].as_numpy()[0], 1000)
        self.assertEqual(triton_datas[3].name(), 'INPUT3')
        self.assertEqual(triton_datas[3].as_numpy()[0], 2000)
        self.assertEqual(triton_datas[4].name(), 'INPUT4')
        self.assertEqual(triton_datas[4].as_numpy()[0].decode('utf-8'), 'mono')

    def test_to_triton_data_error(self):
        towhee_data = [MockTritonPythonBackendTensor('INPUT0', np.array([1]))]
        triton_datas = ToTritonData(towhee_data).get_triton_tensor('INPUT')
        self.assertTrue(triton_datas is None)
