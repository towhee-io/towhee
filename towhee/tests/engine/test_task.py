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
from collections import namedtuple

from towhee.engine.task import Task


def mock_op(k1, k2):
    Point = namedtuple('Point', ['x', 'y'])
    return Point(k1, k2)


class MockOutputsWriter:
    def __init__(self):
        self.outputs = None

    def write(self, task: Task):
        self.outputs = task.outputs._asdict()


class TestTask(unittest.TestCase):
    """
    Task basic test
    """

    def test_task_basic_func(self):
        args = {'arg1': 1, 'arg2': 'test'}
        inputs = {'k1': 1, 'k2': 'v1'}
        mock_output = MockOutputsWriter()
        task = Task('mock_op', 'mock_op', args, inputs, 0)
        task.add_task_finish_handler(mock_output.write)
        self.assertEqual(task.op_name, 'mock_op')
        self.assertEqual(task.inputs, inputs)
        self.assertEqual(task.op_args, args)
        task.execute(mock_op)
        self.assertEqual(task.outputs.x, inputs['k1'])
        self.assertEqual(task.outputs.y, inputs['k2'])
        self.assertEqual(mock_output.outputs, task.outputs._asdict())
        self.assertGreater(task.runtime, 0)
