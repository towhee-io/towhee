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
import torchvision

from .pytorch_video_classification_operator import PyTorchVideoClassificationOperator
from towhee.tests.mock_operators.video_representation_frame.video_representation_frame import VidRepr


class TestVidClass(unittest.TestCase):
    """
    Video classification test
    """
    def test_vid_class(self):
        test_video = Path(__file__).parent / 'video.mp4'
        rep = VidRepr(num=2, inter=2)
        frames = rep(str(test_video))
        model = 'resnet50'
        op = PyTorchVideoClassificationOperator(model)
        f'frames num is {len(frames)}'
        output = op(frames.imgs)
        self.assertEqual((1, 2000), output[0].shape)
