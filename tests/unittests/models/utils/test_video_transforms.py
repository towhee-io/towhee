# Copyright 2022 Zilliz. All rights reserved.
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
import subprocess
import unittest
from towhee.models.utils.video_transforms import VideoTransforms



class TestVideoTransforms(unittest.TestCase):
    """
    Test transforms for pretrained models in pytorchvideo
    """
    def setUp(self) -> None:
        url = 'https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4'
        subprocess.check_call(['wget', url])

    def test_transforms(self):
        tfms_slow_r50 = VideoTransforms('slow_r50')
        out_slow_r50 = tfms_slow_r50('archery.mp4')
        self.assertEqual(out_slow_r50['video'].shape, (3, 8, 256, 256))
        self.assertEqual(out_slow_r50['audio'].shape, (441344,))

        tfms_c2d_r50 = VideoTransforms('slow_r50')
        out_c2d_r50 = tfms_c2d_r50('archery.mp4')
        self.assertEqual(out_c2d_r50['video'].shape, (3, 8, 256, 256))
        self.assertEqual(out_c2d_r50['audio'].shape, (441344,))

        tfms_i3d_r50 = VideoTransforms('slow_r50')
        out_i3d_r50 = tfms_i3d_r50('archery.mp4')
        self.assertEqual(out_i3d_r50['video'].shape, (3, 8, 256, 256))
        self.assertEqual(out_i3d_r50['audio'].shape, (441344,))

        tfms_slowfast_r50 = VideoTransforms('slowfast_r50')
        tfms_slowfast_r101 = VideoTransforms('slowfast_r101')
        out_slowfast_r50 = tfms_slowfast_r50('archery.mp4')
        out_slowfast_r101 = tfms_slowfast_r101('archery.mp4')
        self.assertEqual(out_slowfast_r50['video_slow'].shape, out_slowfast_r101['video_slow'].shape, (3, 8, 256, 256))
        self.assertEqual(out_slowfast_r50['video_fast'].shape, out_slowfast_r101['video_fast'].shape, (3, 32, 256, 256))
        self.assertEqual(out_slowfast_r50['audio'].shape, out_slowfast_r101['audio'].shape, (441344,))

        tfms_x3d_xs = VideoTransforms('x3d_xs')
        tfms_x3d_s = VideoTransforms('x3d_s')
        tfms_x3d_m = VideoTransforms('x3d_m')
        out_x3d_xs = tfms_x3d_xs('archery.mp4')
        out_x3d_s = tfms_x3d_s('archery.mp4')
        out_x3d_m = tfms_x3d_m('archery.mp4')
        self.assertEqual(out_x3d_xs['video'].shape, (3, 4, 182, 182))
        self.assertEqual(out_x3d_s['video'].shape, (3, 13, 182, 182))
        self.assertEqual(out_x3d_m['video'].shape, (3, 16, 256, 256))
        self.assertEqual(out_x3d_xs['audio'].shape, (441344,))
        self.assertEqual(out_x3d_s['audio'].shape, (441344,))
        self.assertEqual(out_x3d_m['audio'].shape, (441344,))

        tfms_mvit_base_16x4 = VideoTransforms('mvit_base_16x4')
        tfms_mvit_base_32x3 = VideoTransforms('mvit_base_32x3')
        out_mvit_base_16x4 = tfms_mvit_base_16x4('archery.mp4')
        out_mvit_base_32x3 = tfms_mvit_base_32x3('archery.mp4')
        self.assertEqual(out_mvit_base_16x4['video'].shape, (3, 16, 224, 224))
        self.assertEqual(out_mvit_base_32x3['video'].shape, (3, 32, 224, 224))
        self.assertEqual(out_mvit_base_16x4['audio'].shape, (441344,))
        self.assertEqual(out_mvit_base_32x3['audio'].shape, (441344,))

        tfms_csn_r101 = VideoTransforms('csn_r101')
        out_csn_r101 = tfms_csn_r101('archery.mp4')
        self.assertEqual(out_csn_r101['video'].shape, (3, 32, 256, 256))
        self.assertEqual(out_csn_r101['audio'].shape, (441344,))

        tfms_r2plus1d_r50 = VideoTransforms('r2plus1d_r50')
        out_r2plus1d_r50 = tfms_r2plus1d_r50('archery.mp4')
        self.assertEqual(out_r2plus1d_r50['video'].shape, (3, 16, 256, 256))
        self.assertEqual(out_r2plus1d_r50['audio'].shape, (441344,))

    def tearDown(self) -> None:
        os.unlink('archery.mp4')


if __name__ == '__main__':
    unittest.main(verbosity=1)
