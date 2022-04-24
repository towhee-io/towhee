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

# import os
# import subprocess
import numpy

import unittest
from towhee.models.utils.video_transforms import transform_video



class TestVideoTransforms(unittest.TestCase):
    """
    Test transforms for pretrained models in pytorchvideo
    """
    def setUp(self) -> None:
        # url = 'https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4'
        # subprocess.check_call(['wget', url])

        self.video = numpy.ndarray((3, 30, 240, 320))
        self.video.dtype = numpy.float32


    def test_transform_base(self):
        out1 = transform_video(video=self.video)
        self.assertEqual(out1['video'].shape, (3, 8, 256, 256))
        out2 = transform_video(video=self.video, side_size=224, crop_size=224, num_frames=16, sampling_rate=8)
        self.assertEqual(out2['video'].shape, (3, 16, 224, 224))

    def test_models(self):
        out_slow_r50 = transform_video(video=self.video, model_name='slow_r50')
        self.assertEqual(out_slow_r50['video'].shape, (3, 8, 256, 256))

        out_c2d_r50 = transform_video(video=self.video, model_name='slow_r50')
        self.assertEqual(out_c2d_r50['video'].shape, (3, 8, 256, 256))

        out_i3d_r50 = transform_video(video=self.video, model_name='slow_r50')
        self.assertEqual(out_i3d_r50['video'].shape, (3, 8, 256, 256))

        out_x3d_xs = transform_video(video=self.video, model_name='x3d_xs')
        out_x3d_s = transform_video(video=self.video, model_name='x3d_s')
        out_x3d_m = transform_video(video=self.video, model_name='x3d_m')
        self.assertEqual(out_x3d_xs['video'].shape, (3, 4, 182, 182))
        self.assertEqual(out_x3d_s['video'].shape, (3, 13, 182, 182))
        self.assertEqual(out_x3d_m['video'].shape, (3, 16, 256, 256))

        out_mvit_base_16x4 = transform_video(video=self.video, model_name='mvit_base_16x4')
        out_mvit_base_32x3 = transform_video(video=self.video, model_name='mvit_base_32x3')
        self.assertEqual(out_mvit_base_16x4['video'].shape, (3, 16, 224, 224))
        self.assertEqual(out_mvit_base_32x3['video'].shape, (3, 32, 224, 224))

        out_csn_r101 = transform_video(video=self.video, model_name='csn_r101')
        self.assertEqual(out_csn_r101['video'].shape, (3, 32, 256, 256))

        out_r2plus1d_r50 = transform_video(video=self.video, model_name='r2plus1d_r50')
        self.assertEqual(out_r2plus1d_r50['video'].shape, (3, 16, 256, 256))

        out_slowfast_r50 = transform_video(video=self.video, model_name='slowfast_r50')
        out_slowfast_r101 = transform_video(video=self.video, model_name='slowfast_r101')
        self.assertEqual(out_slowfast_r50['video_slow'].shape, out_slowfast_r101['video_slow'].shape, (3, 8, 256, 256))
        self.assertEqual(out_slowfast_r50['video_fast'].shape, out_slowfast_r101['video_fast'].shape, (3, 32, 256, 256))

    # def tearDown(self) -> None:
    #     os.unlink('archery.mp4')


if __name__ == '__main__':
    unittest.main(verbosity=1)
