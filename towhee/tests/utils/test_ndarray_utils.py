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
import cv2
import numpy as np
from pathlib import Path

from towhee.types import Image
from towhee.utils.ndarray_utils import from_ndarray, from_src, to_ndarray, rgb2bgr

logo_path = os.path.join(Path(__file__).parent.parent.parent.parent.resolve(), 'towhee_logo.png')


class TestPilUtils(unittest.TestCase):
    """
    Test for ndarray utils.
    """
    def test_from_src(self):
        img_1 = from_src(logo_path)
        img_2 = from_src(Path(logo_path))

        self.assertIsInstance(img_1, Image)
        self.assertIsInstance(img_2, Image)
        self.assertEqual(img_1.image, img_2.image)
        self.assertEqual(img_1.width, img_2.width)
        self.assertEqual(img_1.height, img_2.height)
        self.assertEqual(img_1.channel, img_2.channel)
        self.assertEqual(img_1.mode, img_2.mode)
        self.assertTrue((img_1.array == img_2.array).all())

    def test_from_ndarray(self):
        ndarray_img = cv2.imread(logo_path)
        towhee_img = from_ndarray(ndarray_img, 'BGR')

        self.assertIsInstance(towhee_img, Image)

        self.assertEqual(ndarray_img.shape[0], towhee_img.height)
        self.assertEqual(ndarray_img.shape[1], towhee_img.width)
        self.assertEqual(ndarray_img.shape[2], towhee_img.channel)
        self.assertEqual('BGR', towhee_img.mode)
        self.assertTrue((np.array(ndarray_img) == towhee_img.array).all())

    def test_to_ndarray(self):
        ndarray_img = cv2.imread(logo_path)
        img_bytes = ndarray_img.tobytes()
        img_width = ndarray_img.shape[1]
        img_height = ndarray_img.shape[0]
        img_channel = len(cv2.split(ndarray_img))
        img_mode = 'BGR'
        img_array = ndarray_img

        towhee_img = Image(img_bytes, img_width, img_height, img_channel, img_mode, img_array)
        ndarray_img = to_ndarray(towhee_img)

        self.assertIsInstance(ndarray_img, np.ndarray)

        self.assertEqual(ndarray_img.shape[0], towhee_img.height)
        self.assertEqual(ndarray_img.shape[1], towhee_img.width)
        self.assertEqual(ndarray_img.shape[2], towhee_img.channel)
        self.assertEqual('BGR', towhee_img.mode)
        self.assertTrue((ndarray_img == towhee_img.array).all())

    def test_rgb2bgr(self):
        ndarray_img = cv2.imread(logo_path)
        ndarray_img = cv2.cvtColor(ndarray_img, cv2.COLOR_BGR2RGB)

        img_bytes = ndarray_img.tobytes()
        img_width = ndarray_img.shape[1]
        img_height = ndarray_img.shape[0]
        img_channel = len(cv2.split(ndarray_img))
        img_mode = 'RGB'
        img_array = ndarray_img

        towhee_img = Image(img_bytes, img_width, img_height, img_channel, img_mode, img_array)

        towhee_bgr = rgb2bgr(towhee_img)
        ndarray_bgr = rgb2bgr(ndarray_img)

        self.assertTrue((towhee_bgr == ndarray_bgr).all())
