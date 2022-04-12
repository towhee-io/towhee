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
import towhee.functional.mixins.dataset
import towhee.functional.mixins.display
import towhee.functional.mixins.entity_mixin
import towhee.functional.mixins.metric
import towhee.functional.mixins.parallel
import towhee.functional.mixins.state
from towhee.functional.mixins.display import _ndarray_brief, to_printable_table
from towhee import DataCollection


def load_tests(loader, tests, ignore):
    #pylint: disable=unused-argument
    for mod in [
            towhee.functional.mixins.computer_vision,
            towhee.functional.mixins.dataset,
            towhee.functional.mixins.display,
            towhee.functional.mixins.entity_mixin,
            towhee.functional.mixins.metric,
            towhee.functional.mixins.parallel,
            towhee.functional.mixins.state,
    ]:
        tests.addTests(doctest.DocTestSuite(mod))

    return tests


if __name__ == '__main__':
    unittest.main()


class TestDisplayMixin(unittest.TestCase):
    """
    Unit test for DisplayMixin.
    """

    def test_ndarray_bref(self):
        arr = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
        # pylint: disable=protected-access
        self.assertEqual(_ndarray_brief(arr, 3),
                         '[1.1, 2.2, 3.3, ...] shape=(3, 2)')

    def test_to_printable_table(self):
        dc = DataCollection([[1.1, 2.2], [3.3, 4.4]])
        # pylint: disable=protected-access
        plain_tbl = to_printable_table(dc._iterable, tablefmt='plain')
        self.assertEqual(plain_tbl, '1.1  2.2\n3.3  4.4')

        html_tbl = to_printable_table(dc._iterable, tablefmt='html')
        html_str = '<table style="border-collapse: collapse;"><tr></tr> '\
                   '<tr><td style="text-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3;">1.1</td> '\
                   '<td style="text-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3;">2.2</td></tr> '\
                   '<tr><td style="text-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3;">3.3</td> '\
                   '<td style="text-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3;">4.4</td></tr></table>'
        self.assertEqual(html_tbl, html_str)

        dc = DataCollection([['hello'], ['world']])
        plain_tbl = to_printable_table(dc._iterable, tablefmt='plain')
        self.assertEqual(plain_tbl, 'hello\nworld')

        html_tbl = to_printable_table(dc._iterable, tablefmt='html')
        html_str = '<table style="border-collapse: collapse;"><tr></tr> '\
                   '<tr><td style="text-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3;">hello</td></tr> '\
                   '<tr><td style="text-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3;">world</td></tr></table>'
        self.assertEqual(html_tbl, html_str)

    def test_show(self):
        logo_path = os.path.join(
            Path(__file__).parent.parent.parent.parent.resolve(),
            'towhee_logo.png')
        img = cv2.imread(logo_path)
        towhee_img = Image(img, 'BGR')
        dc = DataCollection([[1, img, towhee_img], [2, img, towhee_img]])
        dc.show(tablefmt='plain')
        dc.show(tablefmt='html')
