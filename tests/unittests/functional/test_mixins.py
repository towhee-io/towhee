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
import doctest
import unittest
import os
import cv2
import numpy as np
from pathlib import Path
from towhee._types.image import Image

import towhee.functional.mixins.computer_vision
import towhee.functional.mixins.entity_mixin
from towhee.functional.mixins.display import _ndarray_brief_repr, to_printable_table
from towhee import DataCollection

for mod in [
        towhee.functional.mixins.computer_vision,
        towhee.functional.mixins.entity_mixin
]:
    TestDataCollectionMixins = doctest.DocTestSuite(mod)
    unittest.TextTestRunner(verbosity=4).run(TestDataCollectionMixins)

if __name__ == '__main__':
    unittest.main()


class TestDisplayMixin(unittest.TestCase):
    """
    Unit test for DisplayMixin.
    """

    def test_ndarray_repr(self):
        arr = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
        # pylint: disable=protected-access
        self.assertEqual(_ndarray_brief_repr(arr, 3), '[1.1, 2.2, 3.3, ...] shape=(3, 2)')

    def test_to_printable_table(self):
        dc = DataCollection([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
        # pylint: disable=protected-access
        to_printable_table(dc._iterable, tablefmt='plain')
        to_printable_table(dc._iterable, tablefmt='html')

        logo_path = os.path.join(Path(__file__).parent.parent.parent.parent.resolve(), 'towhee_logo.png')
        img = cv2.imread(logo_path)
        towhee_img = Image(img, 'BGR')
        dc = DataCollection([[1, img, towhee_img], [2, img, towhee_img]])
        to_printable_table(dc._iterable, tablefmt='plain')
        to_printable_table(dc._iterable, tablefmt='html')

    def test_show(self):
        dc = DataCollection([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
        dc.show(tablefmt='plain')
        dc.show(tablefmt='html')
