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
import sys,os
# sys.path.append(os.getcwd())
from towhee.dag.utils.callstack import Callstack

class TestCallStack(unittest.TestCase):
    def test_init(self):
        s1 = Callstack()
        s2 = Callstack(ignore = 1)
        s3 = Callstack(ignore = 2)
        f1 = s1.frames
        f2 = s2.frames
        self.assertIsInstance(f1, list)
        self.assertIsInstance(f2, list)
        self.assertEqual(s1.size, s2.size + 1)
        self.assertEqual(s1.size, s3.size + 2)

    def test_num_frames(self):
        s1 = Callstack()
        s2 = Callstack(ignore = 3)
        self.assertEqual(s1.size, s1.num_frames())
        self.assertEqual(s2.size, s2.num_frames())
        self.assertEqual(s1.num_frames(), s2.num_frames() + 3)

    def test_find_func(self):
        s = Callstack()
        self.assertEqual(s.find_func('test_find_func'), 0)
        self.assertEqual(s.find_func('<module>'), len(s.frames)-1)
        self.assertIsNone(s.find_func('random_func_name_that_not_exits'))

    def test_hash(self):
        s = Callstack()
        output_1 = s.hash(items = ['code_context'])
        output_2 = s.hash(items = ['code_context'])
        self.assertEqual(output_1,output_2)
        output_3 = s.hash(0, 5, ['filename', 'function', 'position'])
        output_4 = s.hash(0, 5, ['filename', 'function', 'position'])
        self.assertEqual(output_3, output_4)
        output_5 = s.hash(
            items = [
                'filename', 'lineno', 'function', 
                'code_context', 'position', 'lasti'
            ]
        )
        output_6 = s.hash(0, 8, ['filename'])
        self.assertEqual(len(output_5), len(output_6))
        self.assertRaises(ValueError, s.hash, items=['attribute_that_not_exists'])
        self.assertRaises(IndexError, s.hash, -1, 1, items=['filename'])
        self.assertRaises(IndexError, s.hash, 0, 1000, items=['filename'])
        self.assertRaises(IndexError, s.hash, 10, 1, items=['filename'])
                

if __name__ == '__main__':
    unittest.main()