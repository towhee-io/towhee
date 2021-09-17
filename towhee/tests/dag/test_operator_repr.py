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
import yaml
import os

from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr

src = """
-
    name: 'test_op_1'
    function: 'test_function'
    inputs:
        -
            name: 'input_1'
            df: 'test_df_1'
            col: 1
        -
            name: 'input_2'
            df: 'test_df_2'
            col: 1
    outputs:
        -
            name: 'output_1'
            df: 'test_df_1'
            col: 2
    iterators:
        -
            df: 'test_df_1'
            iter:
                type: map
        -
            df: 'test_df_2'
            iter:
                type: map
"""
cur_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(cur_path, 'test_operator.yaml')
essentials = {'name', 'function', 'inputs', 'outputs', 'iterators'}
dataframes = DataframeRepr.from_yaml(os.path.join(cur_path, 'test_dataframe.yaml'))


class TestOperatorRepr(unittest.TestCase):
    """Basic test case for `OperatorRepr`.
    """
    def test_init(self):
        # Create a `OperatorRepr` object
        self.repr = OperatorRepr()
        self.assertTrue(isinstance(self.repr, OperatorRepr))

    def test_is_valid(self):
        # When the information is valid YAML format
        info = yaml.safe_load(src)
        self.assertTrue(OperatorRepr.is_valid(info))

        # When the information is not valid YAML format
        false_src_1 = 'test_src that is not of YAML format'
        false_info_1 = [yaml.safe_load(false_src_1)]
        self.assertFalse(OperatorRepr.is_valid(false_info_1))

        # When the information is a valid YAML format but cannot describe a operator
        false_src_2 = """test_key : 'test_value'"""
        false_info_2 = [yaml.safe_load(false_src_2)]
        self.assertFalse(OperatorRepr.is_valid(false_info_2))

    def test_yaml_import(self):
        self.repr = OperatorRepr.from_yaml(src, dataframes)

        self.assertEqual(self.repr['test_op_1'].name, 'test_op_1')

        # An operator has two dict type properties `inputs` and `outputs`
        self.assertTrue(isinstance(self.repr['test_op_1'].inputs, dict))
        self.assertTrue(isinstance(self.repr['test_op_1'].outputs, dict))

        # In this case, the 'idx' of all the inputs are 1
        self.assertEqual(self.repr['test_op_1'].inputs['input_1']['idx'], 1)
        self.assertEqual(self.repr['test_op_1'].inputs['input_2']['idx'], 1)


if __name__ == '__main__':
    unittest.main()
