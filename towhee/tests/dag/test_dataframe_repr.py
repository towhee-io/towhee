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


from typing import List, NamedTuple, Tuple
import unittest

from towhee.dag.dataframe_repr import DataframeRepr


def test_op_a(a: str, b: int, c: List) -> NamedTuple:
    retval = NamedTuple('data', [('d', List), ('e', Tuple)])
    return retval(c, (a, b))


class TestDataframeRepr(unittest.TestCase):
    """Basic test case for `DataframeRepr`.
    """

    def setUp(self):
        self.repr = DataframeRepr('test')

    def test_input_auto_annotate(self):
        self.repr.from_input_annotations(test_op_a)
        # TODO
        self.assertTrue(isinstance(self.repr['a'], str))

    def test_output_auto_annotate(self):
        # self.repr.from_output_annotations(test_op_a)
        # TODO
        # self.assertTrue(isinstance("", str))
        pass

    def test_serialize(self):
        # yaml = self.repr.serialize()
        # TODO
        # self.assertTrue(isinstance("", str))
        pass


if __name__ == '__main__':
    unittest.main()
