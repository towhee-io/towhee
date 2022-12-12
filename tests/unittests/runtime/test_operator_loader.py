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

from towhee.runtime import ops


class TestOperatorLoader(unittest.TestCase):
    """
    Unit test for operator loader.
    """
    def test_old(self):
        op = ops.local.old_format_op(10)
        self.assertEqual(op(10), 20)

    def test_new(self):
        op = ops.local.new_format_op(10)
        self.assertEqual(op(10), 20)

    def test_invalid_old(self):
        op = ops.local.old_old_format_op_invalid(10)
        with self.assertRaises(RuntimeError) as e:
            op(10)
            self.assertEqual(e.msg, 'Loading operator with error:No module named \'NotExistPackage\'')

    def test_invalid_new(self):
        op = ops.local.new_format_op_invalid(10)
        with self.assertRaises(RuntimeError) as e:
            op(10)
            self.assertEqual(e.msg, 'Loading operator with error:No module named \'NotExistPackage\'')

    def test_inconsistent(self):
        op = ops.local.old_format_op_inconsistent(10)
        with self.assertRaises(RuntimeError) as e:
            op(10)
            self.assertIn('Loading operator with error:[Errno 2] No such file or directory:', e.msg)
