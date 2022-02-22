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

import os
import unittest
import numpy as np

from pathlib import Path
from PIL import Image as PILImage

from towhee.types import Image

logo_path = os.path.join(Path(__file__).parent.parent.parent.parent.resolve(), 'towhee_logo.png')
img = PILImage.open(logo_path)
img_bytes = img.tobytes()
img_width = img.width
img_height = img.height
img_channel = len(img.split())
img_mode = img.mode
img_array = np.array(img)
array_size = np.array(img).shape


class TestImage(unittest.TestCase):
    """
    Test Image class.
    """

    def test_base(self):
        towhee_img = Image(img_bytes, img_width, img_height, img_channel, img_mode, img_array)

        self.assertIsInstance(towhee_img, Image)
        self.assertEqual(towhee_img.width, img_width)
        self.assertEqual(towhee_img.height, img_height)
        self.assertEqual(towhee_img.channel, img_channel)
        self.assertEqual(towhee_img.mode, img_mode)
        self.assertTrue((towhee_img.array == img_array).all())

        self.assertEqual(array_size[0], img_height)
        self.assertEqual(array_size[1], img_width)
        self.assertEqual(array_size[2], img_channel)

        self.assertFalse(towhee_img.key_frame)

        cp_array = towhee_img.array
        cp_array[0][0][0] = 78
        self.assertFalse((towhee_img.array == cp_array).all())
        self.assertTrue((towhee_img.array == img_array).all())

        towhee_img_1 = Image(img_bytes, img_width, img_height, img_channel, img_mode)
        with self.assertRaises(AttributeError):
            _ = towhee_img_1.array
            # self.assertIn('The array of image is not given, please call `Image.to_ndarray()` function to get the ndarray.', log.output[0])

    def test_toarray(self):
        towhee_img = Image(img_bytes, img_width, img_height, img_channel, img_mode)
        array_1 = np.array(img)
        array_2 = towhee_img.to_ndarray()
        array_3 = array_2.reshape(img_height, img_width, img_channel)

        self.assertEqual(array_1.sum(), array_2.sum())
        self.assertTrue((array_1 == array_3).all())


if __name__ == '__main__':
    unittest.main()
