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
import towhee
# pylint: disable=import-outside-toplevel


class TestEntity(unittest.TestCase):
    """
    Unit test for Pandas related functions.
    """

    def test_from_pd(self):
        import pandas as pd

        df = pd.DataFrame(dict(a=range(5), b=range(5)))
        dc = towhee.from_df(df)
        self.assertListEqual(dc.df.a.to_list(), [0, 1, 2, 3, 4])

        dc.df.b = [0, 2, 4, 6, 8]
        self.assertListEqual(dc.df['b'].to_list(), [0, 2, 4, 6, 8])
        self.assertEqual(repr(dc),
                         '   a  b\n0  0  0\n1  1  2\n2  2  4\n3  3  6\n4  4  8')

    def test_dataframe_map_siso(self):
        import pandas as pd

        df = pd.DataFrame(dict(a=range(5), b=range(5)))
        dc = towhee.from_df(df)
        dc = dc.runas_op['a', 'c'](func=lambda x: x + 1)
        self.assertListEqual(dc['c'].to_list(), [1, 2, 3, 4, 5])

    def test_dataframe_map_miso(self):
        import pandas as pd

        df = pd.DataFrame(dict(a=range(5), b=range(5)))
        dc = towhee.from_df(df)
        dc = dc.runas_op[('a', 'b'), 'c'](func=lambda x, y: x + y)
        self.assertListEqual(dc['c'].to_list(), [0, 2, 4, 6, 8])

    def test_dataframe_map_mimo(self):
        import pandas as pd

        df = pd.DataFrame(dict(a=range(5), b=range(5)))
        dc = towhee.from_df(df)
        dc = dc.runas_op[('a', 'b'), ('c', 'd')](func=lambda x, y: (x + y, x - y))
        self.assertListEqual(dc['c'].to_list(), [0, 2, 4, 6, 8])
        self.assertListEqual(dc['d'].to_list(), [0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
