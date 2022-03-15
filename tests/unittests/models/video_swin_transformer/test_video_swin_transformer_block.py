# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import torch

from towhee.models.video_swin_transformer.video_swin_transformer_block import VideoSwinTransformerBlock


class VideoSwinTransformerBlockTest(unittest.TestCase):
    def test_video_swin_transformer_block(self):
        dim = 144
        depth = 3
        window_size = [2, 2, 2]
        num_heads = 8
        stb3d = VideoSwinTransformerBlock(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size)
        b = 3
        d = 4
        h = 4
        w = 4
        c = dim
        input_tensor = torch.rand(b, c, d, h, w)
        out = stb3d(input_tensor)
        self.assertTrue(out.shape == torch.Size([3, 144, 4, 4, 4]))
