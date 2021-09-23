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
import os
import yaml

from towhee.dag.variable_repr import VariableRepr
from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr
from towhee.dag.graph_repr import GraphRepr

src = """
-
    graph:
        name: 'test'

    operators:
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
        -
            name: 'test_op_2'
            function: 'test_function'
            inputs:
                -
                    name: 'output_1'
                    df: 'test_df_1'
                    col: 2
            outputs:
                -
                    name: 'output_2'
                    df: 'test_df_2'
                    col: 2
            iterators:
                -
                    df: 'test_df_1'
                    iter:
                        type: map

    dataframes:
        -
            name: 'test_df_1'
            columns:
                -
                    vtype: 'test_vtype_1'
                    dtype: 'test_dtype_1'
                -
                    vtype: 'test_vtype_2'
                    dtype: 'test_dtype_2'
        -
            name: 'test_df_2'
            columns:
                -
                    vtype: 'test_vtype_3'
                    dtype: 'test_dtype_3'
                -
                    vtype: 'test_vtype_3'
                    dtype: 'test_dtype_3'
                -
                    vtype: 'test_vtype_3'
                    dtype: 'test_dtype_3'
"""

cur_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(cur_path, 'test_graph.yaml')
essentials = {'graph', 'operators', 'dataframes'}


class TestGraphRepr(unittest.TestCase):
    """Basic test cases for `GraphRepr`.
    """
    def test_init(self):
        # Create a `GraphRepr` object
        self.repr = GraphRepr()

        # `operators` and `dataframes` should be dict type
        self.assertTrue(isinstance(self.repr.operators, dict))
        self.assertTrue(isinstance(self.repr.dataframes, dict))

    def test_dfs(self):
        adj_1 = {'a': ['b'], 'b': ['c'], 'c': ['d']}
        flag_1 = {'a': 0, 'b': 0, 'c': 0}
        status, msg = GraphRepr.dfs('a', adj_1, flag_1, [])
        self.assertEqual(status, False)
        self.assertEqual(msg, '')

        adj_2 = {'a': ['b'], 'b': ['c'], 'c': ['a']}
        flag_2 = {'a': 0, 'b': 0, 'c': 0}
        status, msg = GraphRepr.dfs('a', adj_2, flag_2, [])
        self.assertEqual(status, True)
        self.assertEqual(msg, 'The columns [\'a\', \'b\', \'c\'] forms a loop')

    def test_has_loop(self):
        op_path = os.path.join(cur_path, 'loop_operators.yaml')

        self.repr = GraphRepr.from_yaml(src)
        self.assertTrue(isinstance(self.repr, GraphRepr))
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, False)
        self.assertEqual(msg, '')

        self.repr.operators = OperatorRepr.from_yaml(op_path, self.repr.dataframes)
        status, msg = self.repr.has_loop()
        self.assertEqual(status, True)
        self.assertEqual(msg, 'The columns [\'input_2\', \'output_2\', \'output_3\'] forms a loop')

    def has_iso_df(self):
        df_path = os.path.join(cur_path, 'isolated_dataframe.yaml')

        self.repr = GraphRepr.from_yaml(src)
        self.assertTrue(isinstance(self.repr, GraphRepr))
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, False)
        self.assertEqual(msg, '')

        self.repr.dataframes = DataframeRepr.from_yaml(df_path)
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, True)
        self.assertEqual(msg, 'The DAG contains isolated dataframe(s): {\'isolated_df\'}')

    def test_iso_op(self):
        op_path = os.path.join(cur_path, 'isolated_operator.yaml')

        self.repr = GraphRepr.from_yaml(src)
        self.assertTrue(isinstance(self.repr, GraphRepr))
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, False)
        self.assertEqual(msg, '')

        self.repr.operators = OperatorRepr.from_yaml(op_path, self.repr.dataframes)
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, True)
        self.assertEqual(msg, 'The DAG contains isolated operator(s): {\'isoloated_op_1\'}')

    def test_has_isolation(self):
        df_path = os.path.join(cur_path, 'isolated_dataframe.yaml')
        op_path = os.path.join(cur_path, 'isolated_operator.yaml')

        self.repr = GraphRepr.from_yaml(src)
        self.assertTrue(isinstance(self.repr, GraphRepr))
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, False)
        self.assertEqual(msg, '')

        self.repr.dataframes = DataframeRepr.from_yaml(df_path)
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, True)
        self.assertEqual(msg, 'The DAG contains isolated dataframe(s): {\'isolated_df\'}')

        self.repr.operators = OperatorRepr.from_yaml(op_path, self.repr.dataframes)
        status, msg = self.repr.has_isolation()
        self.assertEqual(status, True)
        self.assertEqual(msg, 'The DAG contains isolated dataframe(s): {\'isolated_df\'}.The DAG contains isolated operator(s): {\'isoloated_op_1\'}')

    def test_is_valid(self):
        # When the information is valid YAML format
        info = yaml.safe_load(src)
        self.assertTrue(GraphRepr.is_valid(info))

        # When the information is not valid YAML format
        false_src_1 = 'test_src that is not of YAML format'
        false_info_1 = [yaml.safe_load(false_src_1)]
        self.assertFalse(GraphRepr.is_valid(false_info_1))

        # When the information is a valid YAML format but cannot describe a DAG
        false_src_2 = """test_key : 'test_value'"""
        false_info_2 = [yaml.safe_load(false_src_2)]
        self.assertFalse(GraphRepr.is_valid(false_info_2))

    def test_yaml_import(self):
        self.repr = GraphRepr.from_yaml(src)
        df = self.repr.dataframes
        op = self.repr.operators

        # `operators` and `dataframes` should be dict type
        self.assertTrue(isinstance(op, dict))
        self.assertTrue(isinstance(df, dict))

        # In this case, `df` contains two dataframes, namely 'test_df_1' and 'test_df_2'
        self.assertTrue({'test_df_1', 'test_df_2'}.issubset(set(df.keys())))

        # Dataframes are stored in the form of dict
        # The keys are `str` type, i.e. the name of the dataframe
        # The values are `DataframeRepr` object
        self.assertTrue(isinstance(df['test_df_1'], DataframeRepr))
        self.assertTrue(isinstance(df['test_df_2'], DataframeRepr))

        # In this case, `test_df_1` has two columns, each columns is a `VariableRepr` object
        # The columns are stored in a list
        self.assertTrue(isinstance(df['test_df_1'].columns[0], VariableRepr))
        self.assertTrue(isinstance(df['test_df_1'].columns[1], VariableRepr))

        # In this case, `op` contains two operators, namely `test_op_1` and `test_op_2`
        self.assertTrue({'test_op_1', 'test_op_2'}.issubset(set(op.keys())))

        # Operators are stored in the form of dict
        # The keys are `str` type, i.e. the name of the operator
        # The values are `OperatorRepr` object
        self.assertTrue(isinstance(op['test_op_1'], OperatorRepr))
        self.assertTrue(isinstance(op['test_op_2'], OperatorRepr))

        # Each `OperatorRepr` object has two `dict` attributes `inputs` and `outputs`
        self.assertTrue(hasattr(op['test_op_1'], 'inputs'))
        self.assertTrue(hasattr(op['test_op_2'], 'inputs'))
        self.assertTrue(isinstance(op['test_op_1'].inputs, dict))
        self.assertTrue(isinstance(op['test_op_1'].outputs, dict))

        # In this case, `test_op_1` has two inputs, namely `inputs_1` and `inputs_2`
        self.assertTrue('input_1' in op['test_op_1'].inputs.keys())
        self.assertTrue('input_2' in op['test_op_1'].inputs.keys())

        # Each input is a `dict` that specifies the dataframe and the column
        self.assertTrue('df' in op['test_op_1'].inputs['input_1'].keys())
        self.assertTrue('idx' in op['test_op_1'].inputs['input_1'].keys())
        self.assertTrue(isinstance(op['test_op_1'].inputs['input_1']['df'], DataframeRepr))
        self.assertTrue(isinstance(op['test_op_1'].inputs['input_1']['idx'], int))

    # def test_yaml_export(self):
    #     # TODO
    #     self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
