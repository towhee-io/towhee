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
import requests
from unittest import mock

from towhee.dag.base_repr import BaseRepr
from towhee.dag.graph_repr import GraphRepr
from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr

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

df_src = """
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

op_src = """
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

src_path = os.path.join(cur_path, 'test_graph.yaml')
essentials = {'graph', 'operators', 'dataframes'}

df_src_path = os.path.join(cur_path, 'test_dataframe.yaml')
df_essentials = {'name', 'columns'}

op_src_path = os.path.join(cur_path, 'test_operator.yaml')
op_essentials = {'name', 'function', 'inputs', 'outputs', 'iterators'}


class TestBaseRepr(unittest.TestCase):
    """Basic test case for `DataframeRepr`.
    """
    def test_init(self):
        # Create a `BaseRepr` object
        self.repr = BaseRepr()
        self.assertTrue(isinstance(self.repr, BaseRepr))

    def test_load_file(self):
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(src)
        graph_list = GraphRepr.load_file(src_path)
        # The return values shoud be a list
        self.assertTrue(isinstance(graph_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(graph_list[0], dict))
        # The dict should at least contain the keys listed in set `essentails`
        self.assertTrue(essentials.issubset(set(graph_list[0].keys())))

        with open(df_src_path, 'w', encoding='utf-8') as f:
            f.write(df_src)
        df_list = DataframeRepr.load_file(df_src_path)
        # The return values shoud be a list
        self.assertTrue(isinstance(df_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(df_list[0], dict))
        # The dict should at least contain the keys listed in set `df_essentails`
        self.assertTrue(df_essentials.issubset(set(df_list[0].keys())))

        with open(op_src_path, 'w', encoding='utf-8') as f:
            f.write(op_src)
        op_list = OperatorRepr.load_file(op_src_path)
        # The return values shoud be a list
        self.assertTrue(isinstance(op_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(op_list[0], dict))
        # The dict should at least contain the keys listed in set `op_essentails`
        self.assertTrue(op_essentials.issubset(set(op_list[0].keys())))

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
        graph_list = mock.Mock(return_value=res)
        requests.get = graph_list

        # test how the `load_url` works
        graph_list = DataframeRepr.load_url('test_url')
        # The return values shoud be a list
        self.assertTrue(isinstance(graph_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(graph_list[0], dict))
        # The dict should at least contain the keys listed in set `essentails`
        self.assertTrue(essentials.issubset(set(graph_list[0].keys())))

        # initialize a respoonse obejct and set `src` as the `text` value
        df_res = MyResponse(df_src)
        # replace return value of `requests.get` with `res`
        df_list = mock.Mock(return_value=df_res)
        requests.get = df_list

        # test how the `load_url` works
        df_list = DataframeRepr.load_url('test_url')
        # The return values shoud be a list
        self.assertTrue(isinstance(df_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(df_list[0], dict))
        # The dict should at least contain the keys listed in set `df_essentails`
        self.assertTrue(df_essentials.issubset(set(df_list[0].keys())))

        # initialize a respoonse obejct and set `src` as the `text` value
        op_res = MyResponse(op_src)
        # replace return value of `requests.get` with `res`
        op_list = mock.Mock(return_value=op_res)
        requests.get = op_list

        # test how the `load_url` works
        op_list = GraphRepr.load_url('test_url')
        # The return values shoud be a list
        self.assertTrue(isinstance(op_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(op_list[0], dict))
        # The dict should at least contain the keys listed in set `op_essentails`
        self.assertTrue(op_essentials.issubset(set(op_list[0].keys())))

    def test_load_str(self):
        graph_list = GraphRepr.load_str(src)
        # The return values shoud be a list
        self.assertTrue(isinstance(graph_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(graph_list[0], dict))
        # The dict should at least contain the keys listed in set `essentails`
        self.assertTrue(essentials.issubset(set(graph_list[0].keys())))

        df_list = DataframeRepr.load_str(df_src)
        # The return values shoud be a list
        self.assertTrue(isinstance(df_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(df_list[0], dict))
        # The dict should at least contain the keys listed in set `df_essentails`
        self.assertTrue(df_essentials.issubset(set(df_list[0].keys())))

        op_list = OperatorRepr.load_str(op_src)
        # The return values shoud be a list
        self.assertTrue(isinstance(op_list, list))
        # Inside the list is a series of dict
        self.assertTrue(isinstance(op_list[0], dict))
        # The dict should at least contain the keys listed in set `op_essentails`
        self.assertTrue(op_essentials.issubset(set(op_list[0].keys())))

    def test_load_src(self):
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(src)
        graph_1 = GraphRepr.load_src(src_path)
        graph_2 = GraphRepr.load_src(src)
        self.assertEqual(graph_1, graph_2)

        with open(df_src_path, 'w', encoding='utf-8') as f:
            f.write(df_src)
        df_1 = DataframeRepr.load_src(df_src_path)
        df_2 = DataframeRepr.load_src(df_src)
        self.assertEqual(df_1, df_2)

        with open(op_src_path, 'w', encoding='utf-8') as f:
            f.write(op_src)
        op_1 = OperatorRepr.load_src(op_src_path)
        op_2 = OperatorRepr.load_src(op_src)
        self.assertEqual(op_1, op_2)


if __name__ == '__main__':
    unittest.main()
