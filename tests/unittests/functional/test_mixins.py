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
import numpy as np

import towhee.functional.mixins.computer_vision
import towhee.functional.mixins.entity_mixin
from towhee.functional.mixins.display import _ndarray_to_html_cell

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
        self.assertEqual(_ndarray_to_html_cell(arr), '[1.1, 2.2, 3.3, 4.4, ...] shape=(3, 2)')
