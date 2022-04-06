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
import json
from towhee import Document
from towhee import DataCollection


class TestDocument(unittest.TestCase):
    """
    Unit test for Entity class.
    """

    def test_init(self):
        dc = DataCollection([
            {
                'a': 1,
                'b': {
                    'c': 2
                }
            },
            {
                'a': 2,
                'b': {
                    'c': 3
                }
            },
        ])
        dc = dc.runas_op(func=lambda x: Document(**x)) \
            .runas_op['a', 'c.d'](func=lambda x: x+1) \
            .runas_op['b.c', 'b.d'](func=lambda x: x*2)
        result = json.dumps(dc.to_list())
        expected = json.dumps([
            {
                'a': 1,
                'b': {
                    'c': 2,
                    'd': 4
                },
                'c': {
                    'd': 2
                }
            },
            {
                'a': 2,
                'b': {
                    'c': 3,
                    'd': 6
                },
                'c': {
                    'd': 3
                }
            },
        ])
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
