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
import threading

from towhee.dataframe import DataFrame
from towhee.dataframe.iterators import MapIterator
from towhee.engine.operator_io import create_reader, create_writer
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.operator_runner.generator_runner import GeneratorRunner
from tests.unittests.mock_operators.generator_operator import generator_operator


def run(runner):
    runner.process()


class TestGeneratorRunner(unittest.TestCase):
    """
    GeneratorRunner test
    """

    def _create_test_obj(self):
        input_df = DataFrame('input', [('num', 'int')])
        out_df = DataFrame('output', [('sum', 'int')])
        writer = create_writer('generator', [out_df])
        reader = create_reader(input_df, 'generator', {'num': 0})
        runner = GeneratorRunner('test', 0, 'generator_operator', 'main',
                                 'mock_operators', {'num': 1}, [reader], writer)
        return input_df, out_df, runner

    def test_generator_runner(self):
        input_df, out_df, runner = self._create_test_obj()
        runner.set_op(generator_operator.GeneratorOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        input_df.put({'num': 10})
        input_df.seal()
        t.join()
        runner.join()
        out_df.seal()
        it = MapIterator(out_df, True)
        res = 0
        for item in it:
            self.assertEqual(item[0][0], res)
            res += 1
        self.assertEqual(out_df.size, 10)
        self.assertEqual(runner.status, RunnerStatus.FINISHED)

    def test_generator_runner_with_multidata(self):
        input_df, out_df, runner = self._create_test_obj()
        runner.set_op(generator_operator.GeneratorOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        input_df.put({'num': 10})
        input_df.put({'num': 5})
        input_df.seal()
        t.join()
        runner.join()
        out_df.seal()
        self.assertEqual(out_df.size, 15)
        self.assertEqual(runner.status, RunnerStatus.FINISHED)

    def test_generator_runner_with_multirunners(self):
        input_df_1, out_df_1, runner_1 = self._create_test_obj()
        out_df_2 = DataFrame('output', [('sum', 'int')])
        writer = create_writer('generator', [out_df_2])
        reader = create_reader(out_df_1, 'generator', {'num': 0})
        runner_2 = GeneratorRunner('test', 0, 'generator_operator', 'main',
                                   'mock_operators', {'num': 1}, [reader], writer)

        runner_1.set_op(generator_operator.GeneratorOperator())
        t1 = threading.Thread(target=run, args=(runner_1, ))
        t1.start()

        runner_2.set_op(generator_operator.GeneratorOperator())
        t2 = threading.Thread(target=run, args=(runner_2, ))
        t2.start()

        input_df_1.put({'num': 1})
        input_df_1.put({'num': 2})
        input_df_1.put({'num': 3})
        input_df_1.seal()
        t1.join()
        runner_1.join()

        # In engine, the op_ctx will do it
        out_df_1.seal()
        t2.join()
        runner_2.join()
        out_df_2.seal()

        self.assertEqual(runner_1.status, RunnerStatus.FINISHED)
        self.assertEqual(runner_2.status, RunnerStatus.FINISHED)
        self.assertEqual(out_df_2.size, 4)
        it = MapIterator(out_df_2, True)
        expect = ['1-2', '2-4', '2-5', '2-5']
        index = 0
        for item in it:
            self.assertEqual(item[0][-1].parent_path, expect[index])
            index += 1

    def test_generator_runner_with_error(self):
        input_df, _, runner = self._create_test_obj()
        runner.set_op(generator_operator.GeneratorOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        input_df.put({'num': 'error_data'})
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FAILED)


if __name__ == '__main__':
    unittest.main()
