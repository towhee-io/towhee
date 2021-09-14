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

from towhee.dag.dataframe_repr import DataframeRepr


class TestDataframeRepr(unittest.TestCase):
    """Basic test case for `DataframeRepr`.
    """
    def test_init(self):
        # Create a `DataframeRepr` object from a string in YAML format
        self.repr = DataframeRepr()
        self.assertTrue(isinstance(self.repr, DataframeRepr))

    def test_yaml_import(self):
        src = """
            name: 'test_df_1'
            columns:
                -
                    vtype: 'test_vtype_1'
                    dtype: 'test_dtype_1'
                -
                    vtype: 'test_vtype_2'
                    dtype: 'test_dtype_2'
        """
        self.repr = DataframeRepr.from_yaml(src)
        # The first column in the dataframe has vtype `test_vtype_1` and dtype `test_dtype_1`
        self.assertEqual(self.repr.columns[0].vtype, 'test_vtype_1')
        self.assertEqual(self.repr.columns[0].dtype, 'test_dtype_1')


if __name__ == '__main__':
    unittest.main()
