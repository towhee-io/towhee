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
from towhee.utils.image_utils import from_pil, to_pil

logo_path = os.path.join(Path(__file__).parent.parent.parent.parent.resolve(), 'towhee_logo.png')


class TestPilUtils(unittest.TestCase):
    """
    Test for pil utils.
    """

    def test_from_pil(self):
        pil_img = PILImage.open(logo_path)
        towhee_img = from_pil(pil_img)

        self.assertIsInstance(towhee_img, Image)

        self.assertEqual(pil_img.height, towhee_img.shape[0])
        self.assertEqual(pil_img.width, towhee_img.shape[1])
        self.assertEqual(len(pil_img.split()), towhee_img.shape[2])
        self.assertEqual(pil_img.mode, towhee_img.mode)
        self.assertTrue((np.array(pil_img) == towhee_img).all())

        pil_img2 = to_pil(towhee_img)

        self.assertIsInstance(pil_img2, PILImage.Image)

        self.assertEqual(pil_img2.height, towhee_img.shape[0])
        self.assertEqual(pil_img2.width, towhee_img.shape[1])
        self.assertEqual(len(pil_img2.split()), towhee_img.shape[2])
        self.assertEqual(pil_img2.mode, towhee_img.mode)
        self.assertTrue((np.array(pil_img2) == towhee_img).all())
