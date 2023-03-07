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
from towhee.dc2 import pipe, ops
from towhee.tools import Visualizer

# pylint: disable=protected-access
class TestVisualizer(unittest.TestCase):
    """
    Unit test for Visualizer.
    """
    p0 = (
        pipe.input('path')
    )

    p1 = (
        p0.map('path', 'path', lambda x: x)
    )

    p2 = (
        p0.map('path', 'image', ops.image_decode.cv2())
            .window('image', 'image', 1, 1, lambda x: x)
            .filter(('path', 'image'), ('path', 'image'), 'path', lambda x: True)
    )

    p = p1.concat(p2).output('image')

    def test_show(self):
        v = Visualizer(self.p.dag_repr)
        v.show()

        data = v._get_data()
        self.assertTrue(len(data), 7)
