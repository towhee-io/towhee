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

from towhee._types import Image
import numpy as np


class TestImage(unittest.TestCase):
    """
    tests for Image
    """

    def test_ndarray_methods(self):
        # pylint: disable=protected-access
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        img = Image(data, 'RGB')
        self.assertEqual(img._mode, 'RGB')
        self.assertListEqual(list(img.shape), [3, 4])
        self.assertEqual(img.sum(), 30)

        arr = np.zeros((3, 4))
        img_arr = arr.view(Image)
        self.assertListEqual(list(img_arr.shape), [3, 4])
        self.assertEqual(img_arr.sum(), 0)
        self.assertEqual(img_arr._mode, None)

        red = img[:1]
        self.assertListEqual(list(red.shape), [1, 4])
        self.assertEqual(red._mode, 'RGB')


if __name__ == '__main__':
    unittest.main()
