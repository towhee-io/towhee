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

# pylint:disable=protected-access
import unittest

import torch
from towhee.serve.triton.triton_client import TritonClient
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils


def normal(*args):
    res = list(args)
    return res


def fail(*args):
    raise pb_utils.TritonModelException('Error test.')


pb_utils.InferenceRequest.set_model('normal', normal)
pb_utils.InferenceRequest.set_model('fail', fail)


class TestTritonClient(unittest.TestCase):
    """
    Unit test for TritonClient class.
    """
    def test_init(self):
        tmodel = TritonClient(model_name = 'normal', input_names=['in_a', 'in_b'], output_names=['out_a', 'out_b'])
        self.assertEqual(tmodel._model_name, 'normal')
        self.assertEqual(tmodel._input_names, ['in_a', 'in_b'])
        self.assertEqual(tmodel._output_names, ['out_a', 'out_b'])

    def test_call(self):
        in_1 = torch.arange(4)
        in_2 = torch.arange(4)
        tmodel = TritonClient(model_name='normal', input_names=['in_a', 'in_b'], output_names=['out_a', 'out_b'])
        res = tmodel(in_1, in_2)
        for i, j in zip(res, [in_1, in_2]):
            self.assertTrue(i.equal(j))

        res = tmodel(in_1, in_b=in_2)
        for i, j in zip(res, [in_1, in_2]):
            self.assertTrue(i.equal(j))

        res = tmodel(in_1=in_1, in_b=in_2)
        for i, j in zip(res, [in_1, in_2]):
            self.assertTrue(i.equal(j))

    def test_fail(self):
        in_1 = torch.arange(4)
        in_2 = torch.arange(4)

        tmodel = TritonClient(model_name='fail', input_names=['in_a', 'in_b'], output_names=['out_a', 'out_b'])
        with self.assertRaises(pb_utils.TritonModelException) as e:
            _ = tmodel(in_1, in_2)
            self.assertEqual(e.message(), 'Error test.')
