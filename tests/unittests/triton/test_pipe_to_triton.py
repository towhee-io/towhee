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

import os
import unittest
import importlib
from tempfile import TemporaryDirectory

import numpy as np

import towhee

from towhee.serve.triton.pipe_to_triton import PipeToTriton
from towhee.utils.serializer import to_triton_data, from_triton_data
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils


class TestPipeToTriton(unittest.TestCase):
    """
    test pipeline to triton model
    """
    def test_normal(self):
        p = (
            towhee.pipe.input('nums', 'arr')
            .flat_map('nums', 'num', lambda x: x)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )

        with TemporaryDirectory(dir='./') as root:
            self.assertTrue(PipeToTriton(p.dag_repr, root, 'ut_pipeline', {'parallelism': 4}).process())
            self.assertTrue(os.path.isfile(os.path.join(root, 'ut_pipeline', '1', 'model.py')))
            self.assertTrue(os.path.isfile(os.path.join(root, 'ut_pipeline', '1', 'pipe.pickle')))
            self.assertTrue(os.path.isfile(os.path.join(root, 'ut_pipeline', 'config.pbtxt')))
            module_spec = importlib.util.spec_from_file_location('py_model', os.path.join(root, 'ut_pipeline', '1', 'model.py'))
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            pipe = module.TritonPythonModel()
            pipe.initialize({})

            input_data = ([1, 2, 3], np.random.rand(3, 3))
            triton_input = np.array([[to_triton_data(input_data)]], dtype=np.object_)
            input_tensors = pb_utils.InferenceRequest([pb_utils.Tensor('INPUT0', triton_input)], [], '')

            res = pipe.execute([input_tensors])
            ret = from_triton_data(res[0].output_tensors()[0].as_numpy()[0])
            self.assertEqual(len(ret[0]), 3)
            for index, item in enumerate(ret[0], 1):
                self.assertTrue((item[0] == (input_data[1] + index)).all())
            pipe.finalize()
