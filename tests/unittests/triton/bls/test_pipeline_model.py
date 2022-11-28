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
import os
import json
from tempfile import TemporaryDirectory

import dill as pickle
import numpy as np

import towhee
from towhee.utils.np_format import NumpyArrayDecoder, NumpyArrayEncoder
import towhee.serve.triton.bls.pipeline_model as pipe_model
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils



class TestPipelineTritonModel(unittest.TestCase):
    """
    Test pipeline model
    """
    def test_normal(self):
        p = (
            towhee.pipe.input('nums', 'arr')
            .flat_map('nums', 'num', lambda x: x)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )

        input_data = ([1, 2, 3], np.random.rand(3, 3))

        triton_input = np.array([json.dumps(input_data, cls=NumpyArrayEncoder)], dtype=np.object_)
        input_tensors = pb_utils.InferenceRequest([pb_utils.Tensor('INPUT0', triton_input)], [], '')

        with TemporaryDirectory(dir='./') as root:
            pf = os.path.join(root, 'pipe.pickle')
            with open(pf, 'wb') as f:
                pickle.dump(p.dag_repr, f)

            pipe = pipe_model.TritonPythonModel()
            pipe._load_pipeline(pf)  # pylint: disable=protected-access
            res = pipe.execute([input_tensors])
            ret = json.loads(res[0].output_tensors()[0].as_numpy()[0], cls=NumpyArrayDecoder)
            self.assertEqual(len(ret), 3)
            for index, item in enumerate(ret, 1):
                self.assertTrue((item[0] == (input_data[1] + index)).all())
