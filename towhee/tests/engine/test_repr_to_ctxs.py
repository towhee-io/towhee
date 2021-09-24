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


from towhee.engine._repr_to_ctx import create_ctxs


src = """
name: 'test'
operators:
    -
        name: 'test_op_1'
        function: 'test_function'
        init_args:
            arg1: 1
            arg2: 'test'
        inputs:
            -
                name: 'k1'
                df: 'test_df_1'
                col: 0
        outputs:
            -
                df: 'test_df_2'
        iter_info:
            type: map
    -
        name: 'test_op_2'
        function: 'test_function'
        init_args:
            arg1: 1
            arg2: 'test'
        inputs:
            -
                name: 'k1'
                df: 'test_df_2'
                col: 2
        outputs:
            -
                df: 'test_df_3'
        iter_info:
            type: map
dataframes:
    -
        name: 'test_df_1'
        columns:
            -
                name: 'k1'
                vtype: 'int'
            -
                name: 'k2'
                vtype: 'int'
    -
        name: 'test_df_2'
        columns:
            -
                name: 'k1'
                vtype: 'int'
    -
        name: 'test_df_3'
        columns:
            -
                name: 'k1'
                vtype: 'int'
"""


class TestReprToCtxs(unittest.TestCase):
    """
    Test repr to ctxs
    """

    def test_to_ctxs(self):
        g_ctx, dataframes = create_ctxs(src)
        self.assertEqual(len(g_ctx.op_ctxs), 2)
        self.assertEqual(len(dataframes), 3)
