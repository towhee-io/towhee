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

from towhee.dag.graph_repr import GraphRepr


class TestGraphRepr(unittest.TestCase):
    """Basic test case for `DataframeRepr`.
    """

    def setUp(self):
        self.repr = GraphRepr('test')

    def test_yaml_import(self):
        # TODO
        # self.assertTrue(True)
        pass

    def test_yaml_export(self):
        # TODO
        # self.assertTrue(True)
        pass


if __name__ == '__main__':
    unittest.main()
