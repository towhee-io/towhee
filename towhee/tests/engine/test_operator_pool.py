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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
from pathlib import Path

from towhee.dag.dataframe_repr import DataframeRepr
from towhee.engine.operator_pool import OperatorPool
from towhee.engine.task import Task


class TestOperatorPool(unittest.TestCase):
    """Basic test case for `OperatorPool`.
    """

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._op_pool = OperatorPool(cache_path=cache_path)

    def test_init(self):
        # The list of available ops should be empty upon initialization.
        self.assertFalse(self._op_pool.available_ops)

    def test_loader(self):
        op_func = 'mock_operators/add_operator'
        task = Task('test', op_func, (), 0)
        self._op_pool.acquire_op(task, args={"factor": 0})


if __name__ == '__main__':
    unittest.main()
