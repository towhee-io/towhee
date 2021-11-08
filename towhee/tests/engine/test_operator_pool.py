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
from pathlib import Path
# from shutil import rmtree

from towhee.operator import Operator
from towhee.engine.operator_pool import OperatorPool
from towhee.engine.task import Task
from towhee.tests import CACHE_PATH
from towhee.hub.file_manager import FileManagerConfig, FileManager


class TestOperatorPool(unittest.TestCase):
    """Basic test case for `OperatorPool`.
    """
    @classmethod
    def setUpClass(cls):
        new_cache = (CACHE_PATH/'test_cache')
        print(new_cache)
        pipeline_cache = (CACHE_PATH/'test_util')
        operator_cache = (CACHE_PATH/'mock_operators')
        fmc = FileManagerConfig()
        fmc.change_default_cache(new_cache)
        pipelines = list(pipeline_cache.rglob('*.yaml'))
        operators = [f for f in operator_cache.iterdir() if f.is_dir()]
        fmc.cache_local_pipeline(pipelines)
        fmc.cache_local_operator(operators)
        fm = FileManager(fmc) # pylint: disable=unused-variable

    # @classmethod
    # def tearDownClass(cls):
    #     new_cache = (CACHE_PATH/'test_cache')
    #     rmtree(str(new_cache))

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._op_pool = OperatorPool(cache_path=cache_path)

    def test_init(self):
        # The list of available ops should be empty upon initialization.
        self.assertFalse(self._op_pool.available_ops)

    def test_acquire_release(self):

        hub_op_id = 'local/add_operator'
        task = Task('test', hub_op_id, {'factor': 0}, (1), 0)

        # Acquire the operator.
        op = self._op_pool.acquire_op(task)

        # Perform some simple operations.
        self.assertTrue(isinstance(op, Operator))
        self.assertEqual(op(1).sum, 1)

        # Release and re-acquire the operator.
        self._op_pool.release_op(op)
        op = self._op_pool.acquire_op(task)

        # Perform more operations.
        self.assertEqual(op(-1).sum, -1)
        self.assertEqual(op(100).sum, 100)


if __name__ == '__main__':
    unittest.main()
