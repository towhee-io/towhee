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
from towhee.dag.utils.callstack import Callstack

cur_dir = os.path.abspath(__file__)


def collect_frames(ignore: int = 0):
    stack = Callstack(ignore)
    stack.frames = [f for f in stack.frames if f.filename == cur_dir]
    stack.size = len(stack.frames)
    return stack


def generate_callstack(ignore: int = 0):
    return collect_frames(ignore)


# create a callstack contains 3 frames
# the structure of the callstack (from top to bottom):
# `collect_frames` frame, `generate_callstack` frame, `<module>` frame
s1 = generate_callstack()
# create a callstack contains 2 frames
s2 = generate_callstack(ignore=1)
# create a callstack contains 1 frames
s3 = generate_callstack(ignore=2)
# create a callstack contains 0 frames
s4 = generate_callstack(ignore=3)


class TestCallStack(unittest.TestCase):
    """
    Unit tests for Callstack class.

    We do not initialize the Callstack directly. Instead, we create two functions
    `generate_frames` and `collect_frames` to initialize the Callstack as well as
    block the frames produced by the unittest. We do not care about the process of
    tests, we want to focus only on the architectures that we designed, which in this
    case are `test_xxx`, `generate_callstack` and `collect_frames`. In this way, the
    structure and depth of the callstack is stable and predictable.
    For convenience, we collect and returns the frames inside the callsatck in the
    the form of list.
    """
    def test_init(self):
        # the callstacks are instances of Callstack class
        self.assertIsInstance(s1, Callstack)
        self.assertIsInstance(s2, Callstack)
        self.assertIsInstance(s3, Callstack)
        self.assertIsInstance(s4, Callstack)

        # the frames should be stored in a list
        self.assertIsInstance(s1.frames, list)
        self.assertIsInstance(s2.frames, list)
        self.assertIsInstance(s3.frames, list)
        self.assertIsInstance(s4.frames, list)

    def test_num_frames(self):
        # assert the size of the callstacks, namely 3, 2, 1, 0
        self.assertEqual(s1.num_frames(), 3)
        self.assertEqual(s2.num_frames(), 2)
        self.assertEqual(s3.num_frames(), 1)
        self.assertEqual(s4.num_frames(), 0)

    def test_find_func(self):
        # create a callstack contain 3 frames
        s = generate_callstack()

        # the `test_find_func` is the first function to be called in this case
        # so it's in the bottom of the current callstack, the index is 2
        self.assertEqual(s.find_func('test_find_func'), 2)
        # the `generate_callstack` is the secondly called function
        self.assertEqual(s.find_func('generate_callstack'), 1)
        # the `collect_frames` is the last function
        self.assertEqual(s.find_func('collect_frames'), 0)
        # the function `func_that_not_exist` does not exist in current callstack
        # therefore return None
        self.assertIsNone(s.find_func('func_that_not_exist'))

    def test_hash(self):
        # create a callstack contain 3 frames
        s = generate_callstack()

        # `out_1` and `out_2` have the same range of frames and items
        # hence the same hash value
        out_1 = s.hash(items=['code_context'])
        out_2 = s.hash(items=['code_context'])
        self.assertEqual(out_1, out_2)

        # `out_3` and `out_3` have the same range of frames and items
        # hence the same hash value
        out_3 = s.hash(0, 2, ['filename', 'function', 'position'])
        out_4 = s.hash(0, 2, ['filename', 'function', 'position'])
        self.assertEqual(out_3, out_4)

        # though `out_5` and `out_6` should be different, the lengths are the same
        out_5 = s.hash(items=['filename', 'lineno', 'function', 'code_context', 'position', 'lasti'])
        out_6 = s.hash(0, 1, ['filename'])
        self.assertEqual(len(out_5), len(out_6))

        # raise ValueError when given unsupported item
        self.assertRaises(ValueError, s.hash, items=['attribute_that_not_exists'])
        # raise IndexError when give invalid range of frames
        self.assertRaises(IndexError, s.hash, -1, 1, items=['filename'])
        self.assertRaises(IndexError, s.hash, 0, 1000, items=['filename'])
        self.assertRaises(IndexError, s.hash, 10, 1, items=['filename'])


if __name__ == '__main__':
    unittest.main()
