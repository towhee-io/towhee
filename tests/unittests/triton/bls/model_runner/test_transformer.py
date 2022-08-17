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
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
from towhee.serve.triton.bls.model_runner.transformer import RequestToOpInputs, OpOutputToResponses


class TestToTowheeData(unittest.TestCase):
    '''
    Test ToTowheeData
    '''

    def test_to_towhee_data_image(self):
        data = pb_utils.InferenceRequest(
            [pb_utils.Tensor('INPUT0', np.array([10])),
             pb_utils.Tensor('INPUT1', np.random.randn(224, 224, 3)),
             pb_utils.Tensor('INPUT2', np.array(['RGB'.encode('utf-8')], dtype=np.object_))]
        )

        schema = [(int, (1,)), (Image, (-1, -1, 3))]
        data = RequestToOpInputs(data, schema).get_towhee_data()
        self.assertEqual(len(data), 2)
        self.assertTrue(isinstance(data[0], int))
        self.assertEqual(data[0], 10)

        self.assertTrue(isinstance(data[1], Image))
        self.assertEqual(data[1].mode, 'RGB')
        self.assertEqual(data[1].shape, (224, 224, 3))

    def test_to_towhee_data_audioframe(self):
        data = pb_utils.InferenceRequest(
            [pb_utils.Tensor('INPUT0', np.random.randn(224, 224, 3)),
             pb_utils.Tensor('INPUT1', np.array([1000])),
             pb_utils.Tensor('INPUT2', np.array([2000])),
             pb_utils.Tensor('INPUT3', np.array(['mono'.encode('utf-8')], dtype=np.object_)),
             pb_utils.Tensor('INPUT4', np.array([10.5]))]
        )

        schema = [(AudioFrame, (-1, -1, 3)), (float, (1,))]
        data = RequestToOpInputs(data, schema).get_towhee_data()
        self.assertEqual(len(data), 2)

        self.assertTrue(isinstance(data[0], AudioFrame))
        self.assertEqual(data[0].layout, 'mono')
        self.assertEqual(data[0].shape, (224, 224, 3))
        self.assertEqual(data[0].timestamp, 2000)
        self.assertEqual(data[0].sample_rate, 1000)

        self.assertTrue(isinstance(data[1], float))
        self.assertEqual(data[1], 10.5)

    def test_to_towhee_data_videoframe(self):
        data = pb_utils.InferenceRequest(
            [pb_utils.Tensor('INPUT0', np.random.randn(224, 224, 3)),
             pb_utils.Tensor('INPUT1', np.array(['RGB'.encode('utf-8')], dtype=np.object_)),
             pb_utils.Tensor('INPUT2', np.array([1000])),
             pb_utils.Tensor('INPUT3', np.array([1])),
             pb_utils.Tensor('INPUT4', np.array([1, 2]))]
        )

        schema = [(VideoFrame, (-1, -1, 3)), (np.int32, (-1,))]
        data = RequestToOpInputs(data, schema).get_towhee_data()
        self.assertEqual(len(data), 2)

        self.assertTrue(isinstance(data[0], VideoFrame))
        self.assertEqual(data[0].mode, 'RGB')
        self.assertEqual(data[0].shape, (224, 224, 3))
        self.assertEqual(data[0].timestamp, 1000)
        self.assertEqual(data[0].key_frame, 1)
        self.assertEqual(data[1].tolist(), [1, 2])

    def test_to_towhee_data_error(self):
        data = pb_utils.InferenceRequest(
            [pb_utils.Tensor('INPUT0', np.array([1]))]
        )

        schema = [(pb_utils.Tensor, (1, ))]
        data = RequestToOpInputs(data, schema).get_towhee_data()
        self.assertTrue(data is None)


class TestToTritonData(unittest.TestCase):
    '''
    Test ToTritonData.
    '''

    def test_to_triton_data_image(self):
        towhee_data = [Image(np.random.randn(224, 224, 3), 'RGB'), 100]
        triton_datas = OpOutputToResponses(towhee_data).get_triton_tensor()
        self.assertEqual(len(triton_datas), 3)
        self.assertEqual(triton_datas[0].name(), 'OUTPUT0')
        self.assertEqual(triton_datas[0].as_numpy().shape, (224, 224, 3))
        self.assertEqual(triton_datas[1].name(), 'OUTPUT1')
        self.assertEqual(triton_datas[1].as_numpy()[0].decode('utf-8'), 'RGB')
        self.assertEqual(triton_datas[2].name(), 'OUTPUT2')
        self.assertEqual(triton_datas[2].as_numpy()[0], 100)

    def test_to_triton_data_videoframe(self):
        towhee_data = [32.5, 'BGR', VideoFrame(np.random.randn(224, 224, 3), 'RGB', 10000, 1)]
        triton_datas = OpOutputToResponses(towhee_data).get_triton_tensor()
        self.assertEqual(len(triton_datas), 6)
        self.assertEqual(triton_datas[0].name(), 'OUTPUT0')
        self.assertEqual(triton_datas[0].as_numpy()[0], 32.5)
        self.assertEqual(triton_datas[1].name(), 'OUTPUT1')
        self.assertEqual(triton_datas[1].as_numpy()[0].decode('utf-8'), 'BGR')
        self.assertEqual(triton_datas[2].name(), 'OUTPUT2')
        self.assertEqual(triton_datas[2].as_numpy().shape, (224, 224, 3))
        self.assertEqual(triton_datas[3].name(), 'OUTPUT3')
        self.assertEqual(triton_datas[3].as_numpy()[0].decode('utf-8'), 'RGB')
        self.assertEqual(triton_datas[4].name(), 'OUTPUT4')
        self.assertEqual(triton_datas[4].as_numpy()[0], 10000)
        self.assertEqual(triton_datas[5].name(), 'OUTPUT5')
        self.assertEqual(triton_datas[5].as_numpy()[0], 1)

    def test_to_triton_data_audioframe(self):
        towhee_data = [np.random.randn(3, 4), AudioFrame(np.random.randn(224, 224, 3), 1000, 2000, 'mono')]
        respones = OpOutputToResponses(towhee_data).to_triton_responses()
        triton_datas = respones.output_tensors()

        self.assertEqual(len(triton_datas), 5)
        self.assertEqual(triton_datas[0].name(), 'OUTPUT0')
        self.assertEqual(triton_datas[0].as_numpy().shape, (3, 4))
        self.assertEqual(triton_datas[1].name(), 'OUTPUT1')
        self.assertEqual(triton_datas[1].as_numpy().shape, (224, 224, 3))
        self.assertEqual(triton_datas[2].name(), 'OUTPUT2')
        self.assertEqual(triton_datas[2].as_numpy()[0], 1000)
        self.assertEqual(triton_datas[3].name(), 'OUTPUT3')
        self.assertEqual(triton_datas[3].as_numpy()[0], 2000)
        self.assertEqual(triton_datas[4].name(), 'OUTPUT4')
        self.assertEqual(triton_datas[4].as_numpy()[0].decode('utf-8'), 'mono')

    def test_to_triton_data_error(self):
        towhee_data = [pb_utils.Tensor('OUTPUT0', np.array([1]))]
        respones = OpOutputToResponses(towhee_data).to_triton_responses()
        self.assertTrue(respones.has_error())
