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

from pathlib import Path
import queue
import unittest


from towhee.serve.triton.bls.model_runner import generator_model
from towhee.serve.triton.bls.caller import local_caller


class TestMapModel(unittest.TestCase):
    '''
    Map model test
    '''
    def test_generator_model(self):
        model = generator_model.TritonPythonModel()
        model._op_config_file = str(Path(__file__).parent.resolve() / 'generator_model_config.json')  # pylint: disable=protected-access
        model.initialize({'model_instance_kind': 'CPU'})

        q = queue.Queue()
        caller = local_caller.LocalStreamCaller({'test_model': model})
        caller.start_stream(q, [(int, (1, )), (int, (1, ))])
        caller.async_stream_call('test_model', [5])
        count = 0
        while True:
            ret = q.get()
            if ret is None:
                break
            self.assertEqual([count, count * 500], ret)
            count += 1
        model.finalize()
