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
from towhee.utils.pil_utils import from_pil, from_src, to_pil

logo_path = os.path.join(Path(__file__).parent.parent.parent.parent.resolve(), 'towhee_logo.png')


class TestPilUtils(unittest.TestCase):
    """
    Test for pil utils.
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

    def test_from_pil(self):
        pil_img = PILImage.open(logo_path)
        towhee_img = from_pil(pil_img)

        self.assertIsInstance(towhee_img, Image)

        self.assertEqual(pil_img.width, towhee_img.width)
        self.assertEqual(pil_img.height, towhee_img.height)
        self.assertEqual(len(pil_img.split()), towhee_img.channel)
        self.assertEqual(pil_img.mode, towhee_img.mode)
        self.assertTrue((np.array(pil_img) == towhee_img.array).all())

    def test_to_pil(self):
        towhee_img = from_src(logo_path)
        pil_img = to_pil(towhee_img)

        self.assertIsInstance(pil_img, PILImage.Image)

        self.assertEqual(pil_img.width, towhee_img.width)
        self.assertEqual(pil_img.height, towhee_img.height)
        self.assertEqual(len(pil_img.split()), towhee_img.channel)
        self.assertEqual(pil_img.mode, towhee_img.mode)
        self.assertTrue((np.array(pil_img) == towhee_img.array).all())
