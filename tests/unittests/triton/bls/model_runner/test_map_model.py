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
import unittest


from towhee.serve.triton.bls.model_runner import map_model
from towhee.serve.triton.bls.caller import local_caller
from towhee.types import Image


class TestMapModel(unittest.TestCase):
    def test_map_model(self):
        model = map_model.TritonPythonModel()
        model._op_config_file = str(Path(__file__).parent.resolve() / 'map_model_config.json')  # pylint: disable=protected-access
        model.initialize({'model_instance_kind': 'CPU'})

        callser = local_caller.LocalCaller({'test_model': model})
        callser.call_model('test_model', ['/test/path'], [(Image, (-1, -1, 3))])
