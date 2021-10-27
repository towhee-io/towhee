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

from .video_representation_frame import VidRepr


class TestVidRepr(unittest.TestCase):
    """
    Video representation test
    """

    def test_vid_repre(self):
        test_video = Path(__file__).parent / 'video.mp4'
        rep = VidRepr()
        out = rep(str(test_video))

        self.assertEqual(len(out.imgs), 1)
