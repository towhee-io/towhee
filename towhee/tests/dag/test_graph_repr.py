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
import requests
from unittest import mock

from towhee.dag.variable_repr import VariableRepr
from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr
from towhee.dag.graph_repr import GraphRepr

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
"""
cur_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(cur_path, 'test_graph.yaml')
essentials = {'graph', 'operators', 'dataframes'}


class TestGraphRepr(unittest.TestCase):
    """Basic test cases for `GraphRepr`.
    """
    def test_init(self):
        # Instantiate a graph representation
        self.repr = GraphRepr()

        # `operators` and `dataframes` should be dict type
        self.assertTrue(isinstance(self.repr.operators, dict))
        self.assertTrue(isinstance(self.repr.dataframes, dict))

    def test_is_format(self):
        # When the information is valid YAML format
        info = yaml.safe_load(src)
        self.assertIsNone(GraphRepr.is_format(info))

        # When the information is not valid YAML format
        false_src_1 = 'test_src that is not of YAML format'
        false_info_1 = yaml.safe_load(false_src_1)
        self.assertRaises(ValueError, GraphRepr.is_format, false_info_1)

        # When the information is a valid YAML format but cannot describe a DAG
        false_src_2 = """test_key : 'test_value'"""
        false_info_2 = yaml.safe_load(false_src_2)
        self.assertRaises(ValueError, GraphRepr.is_format, false_info_2)

    def test_load_file(self):
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(src)
        dict_ = GraphRepr.load_src(src_path)
        # The return value should be a dict
        self.assertTrue(isinstance(dict_, dict))
        # The dict should at least contain the keys in set `essentials`
        self.assertTrue(essentials.issubset(set(dict_.keys())))

    def test_load_url(self):
        # Create a `MyResponse` class for mock to replace the return value of `requests.get`
        class MyResponse:
            def __init__(self, text):
                self._text = text

            @property
            def text(self):
                return self._text

        # initialize a respoonse obejct and set `src` as the `text` value
        res = MyResponse(src)
        # replace return value of `requests.get` with `res`
        list_1 = mock.Mock(return_value=res)
        requests.get = list_1

        # test how the `load_url` works
        dict_ = GraphRepr.load_url('test_url')
        # The return value should be a dict
        self.assertTrue(isinstance(dict_, dict))
        # The dict should at least contain the keys in set `essentials`
        self.assertTrue(essentials.issubset(set(dict_.keys())))

    def test_load_str(self):
        dict_ = GraphRepr.load_str(src)
        # The return value should be a dict
        self.assertTrue(isinstance(dict_, dict))
        # The dict should at least contain the keys in set `essentials`
        self.assertTrue(essentials.issubset(set(dict_.keys())))

    def test_load_src(self):
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(src)
        dict_1 = GraphRepr.load_src(src_path)
        dict_2 = yaml.safe_load(src)
        self.assertEqual(dict_1, dict_2)

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
