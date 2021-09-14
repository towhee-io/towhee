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

from towhee.dag.variable_repr import VariableRepr
from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr
import unittest

from towhee.dag.graph_repr import GraphRepr


class TestGraphRepr(unittest.TestCase):
    """Basic test cases for `GraphRepr`.
    """
    def test_init(self):
        # Instantiate a graph representation
        self.repr = GraphRepr()

        # `operators` and `dataframes` should be dict type
        self.assertTrue(isinstance(self.repr.operators, dict))
        self.assertTrue(isinstance(self.repr.dataframes, dict))

    def test_yaml_import(self):
        src = """
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
                            vtype: 'test_vtype_4'
                            dtype: 'test_dtype_4'
        """
        # self.repr = GraphRepr('test')
        self.repr = GraphRepr.from_yaml(src)
        df = self.repr.dataframes
        op = self.repr.operators

        # `operators` and `dataframes` should be dict type
        self.assertTrue(isinstance(op, dict))
        self.assertTrue(isinstance(df, dict))

        # In this case, `df` contains two dataframes, namely 'test_df_1' and 'test_df_2'
        self.assertTrue('test_df_1' in df.keys())
        self.assertTrue('test_df_2' in df.keys())

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
        self.assertTrue('test_op_1' in op.keys())
        self.assertTrue('test_op_2' in op.keys())

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
