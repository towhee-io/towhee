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
from towhee.engine.operator_runner.runner_base import RunnerBase, RunnerStatus


class MockRunner(RunnerBase):
    def process_step(self) -> bool:
        self._set_finished()
        return True


class TestRunnerBase(unittest.TestCase):
    """
    Runner base test
    """
    def test_runner_base(self):
        mock_runner = MockRunner('mock', 0, 'add_operator', 'main', 'mock_operators', {'num': 1})
        self.assertEqual(mock_runner.status, RunnerStatus.IDLE)
        self.assertTrue(mock_runner.is_idle())
        mock_runner.process()
        self.assertEqual(mock_runner.status, RunnerStatus.FINISHED)
        self.assertEqual(mock_runner.op_name, 'add_operator')
        self.assertEqual(mock_runner.hub_op_id, 'mock_operators')
        self.assertEqual(mock_runner.op_args, {'num': 1})
        self.assertEqual(mock_runner.tag, 'main')

if __name__ == '__main__':
    unittest.main()
