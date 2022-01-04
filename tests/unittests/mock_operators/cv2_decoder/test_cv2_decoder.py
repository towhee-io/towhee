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
from pathlib import Path

from .cv2_decoder import Cv2Decoder


class TestCv2Decoder(unittest.TestCase):
    """
    cv2 decoder test
    """

    def test_cv2_decoder(self):
        test_video = Path(__file__).parent / 'test_video.avi'
        decoder = Cv2Decoder()
        out = decoder(str(test_video))

        self.assertEqual(len(out.imgs), 85)
