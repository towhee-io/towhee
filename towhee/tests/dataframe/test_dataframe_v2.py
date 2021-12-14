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
import time

from towhee.array import Array
from towhee.dataframe.dataframe_v2 import DataFrame
from towhee.iterators.map_iterator import MapIterator


class TestDataframe(unittest.TestCase):
    """
    Test dataframe basic function
    """

    def test_constructors(self):

        def get_columns():
            return [('digit', int), ('letter', str)]

        def get_tuples():
            return [(0, 'a'), (1, 'b'), (2, 'c')]

        def get_arrays():
            return [Array([0, 1, 2]), Array(['a', 'b', 'c'])]

        def get_dict():
            return {('digit', int): Array([0, 1, 2]), ('letter', str): Array(['a', 'b', 'c'])}

        def check_data(df):
            for i in range(3):
                self.assertEqual(df['digit'][i], i)
                self.assertEqual(df['letter'][i], chr(ord('a') + i))
                self.assertEqual(df[i][0], i)
                self.assertEqual(df[i][1], chr(ord('a') + i))
            for i, row in enumerate(df.iter()):
                self.assertEqual(row[0], i)
                self.assertEqual(row[1], chr(ord('a') + i))

        # from list[tuple]
        data = get_tuples()
        columns = get_columns()
        df = DataFrame(columns, name = 'my_df', data = data)

        df.put([3,'d'])
        df.put((4, 'e'))
        df.put({'digit': 5, 'letter': 'f'})
        df.seal()

        it = MapIterator(df)
        it2 = MapIterator(df)

        for i, row in enumerate(it):
            print('row, (values, ): ', i, row)
            print('\nDF Values:\n')
            print(df)
        
        for i, row in enumerate(it2):
            print('row, (values, ): ', i, row)
            print('\nDF Values:\n')
            print(df)
        
       
        # check_data(df)
        
        # # from list[towhee.Array]
        # data = get_arrays()
        # columns = get_columns()
        # df = DataFrame(columns, name = 'my_df', data = data)
        # df.seal()
        # check_data(df)

        # # from dict[str, towhee.Array]
        # data = get_dict()
        # df = DataFrame(None, name = 'my_df', data = data)
        # print(df)

        # df.seal()
        # check_data(df)
        # for x in range(len(df)):


if __name__ == '__main__':
    unittest.main()
