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

src = """
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
src_path = os.path.join(cur_path, 'test_dataframe.yaml')
essentials = {'name', 'columns'}


class TestDataframeRepr(unittest.TestCase):
    """Basic test case for `DataframeRepr`.
    """
    def test_init(self):
        # Create a `DataframeRepr` object
        self.repr = DataframeRepr()
        self.assertTrue(isinstance(self.repr, DataframeRepr))

    def test_is_valid(self):
        # When the information is valid YAML format
        info = yaml.safe_load(src)
        self.assertTrue(DataframeRepr.is_valid(info))

        # When the information is not valid YAML format
        false_src_1 = 'test_src that is not of YAML format'
        false_info_1 = [yaml.safe_load(false_src_1)]
        self.assertFalse(DataframeRepr.is_valid(false_info_1))

        # When the information is a valid YAML format but cannot describe a dataframe
        false_src_2 = """test_key : 'test_value'"""
        false_info_2 = [yaml.safe_load(false_src_2)]
        self.assertFalse(DataframeRepr.is_valid(false_info_2))

    def test_yaml_import(self):
        df = DataframeRepr.from_yaml(src)

        # The dataframes have a list type property `columns`
        self.assertTrue(isinstance(df['test_df_1'].columns, list))
        self.assertTrue(isinstance(df['test_df_2'].columns, list))

        # The first column in `test_df_1` has vtype `test_vtype_1` and dtype `test_dtype_1`
        self.assertEqual(df['test_df_1'].columns[0].vtype, 'test_vtype_1')
        self.assertEqual(df['test_df_1'].columns[0].dtype, 'test_dtype_1')


if __name__ == '__main__':
    unittest.main()
