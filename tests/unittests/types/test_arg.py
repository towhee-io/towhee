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

import numpy as np

from towhee.types import arg, to_image_color, Image


@arg(0, to_image_color('BGR'))
def any2bgr(img):
    return img


@arg(0, to_image_color('rgb'))
def any2rgb(img):
    return img


@arg(0, to_image_color('gbr'))
def bad_cvt(img):
    return img


class TestConvertImageColor(unittest.TestCase):
    """
    tests for arg.convert_image_color
    """

    def test_convert_color(self):
        img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        rgb = Image(img, 'RGB')
        bgr = any2bgr(rgb)
        rgb2 = any2rgb(bgr)
        self.assertEqual(rgb.mode, 'RGB')
        self.assertEqual(bgr.mode, 'BGR')
        self.assertEqual(rgb2.mode, 'RGB')
        self.assertTrue(np.array_equal(rgb, rgb2))

    def test_exception(self):
        img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        rgb = Image(img, 'RGB')

        with self.assertRaises(ValueError):
            bad_cvt(rgb)


if __name__ == '__main__':
    unittest.main()
