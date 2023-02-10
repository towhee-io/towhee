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


# from towhee._types import Image as LegacyImage
import towhee
from towhee.types import Image
from towhee._types import Image as LegacyImage


class TestImage(unittest.TestCase):
    """
    Test Image class.
    """
    def towhee_image(self, ImageClass):  # pylint: disable=invalid-name

        img_height = 600
        img_width = 400
        img_channel = 3
        img_mode = 'RGB'

        img_array = np.random.rand(img_height, img_width, img_channel)
        array_size = img_array.shape
        towhee_img = ImageClass(img_array, img_mode)

        self.assertEqual(towhee_img.width, img_width)
        self.assertEqual(towhee_img.height, img_height)
        self.assertEqual(towhee_img.channel, img_channel)
        self.assertEqual(towhee_img.mode, img_mode)
        self.assertTrue((towhee_img == img_array).all())

        self.assertEqual(array_size[0], img_height)
        self.assertEqual(array_size[1], img_width)
        self.assertEqual(array_size[2], img_channel)

        self.assertTrue((towhee_img == img_array).all())

    def test_image(self):
        self.towhee_image(Image)
        self.towhee_image(LegacyImage)
        self.assertEqual(towhee.types.Image, towhee._types.Image)  # pylint: disable=protected-access


if __name__ == '__main__':
    unittest.main()
