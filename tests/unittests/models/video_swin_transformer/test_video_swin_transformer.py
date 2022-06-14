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

from towhee.models.video_swin_transformer import video_swin_transformer


class VideoSwinTransformerTest(unittest.TestCase):
    def test_video_swin_transformer(self):
        embed_dim = 72
        depths = [3, 3, 6, 3]
        num_heads = [8, 16, 32, 64]
        window_size = [2, 2, 2]
        md = video_swin_transformer.create_model(embed_dim=embed_dim, depths=depths,
                                                 num_heads=num_heads, window_size=window_size)
        input_tensor = torch.rand(1, 3, 4, 4, 4)
        # torch.Size([144, 576, 1, 1, 1])
        out = md(input_tensor)
        self.assertTrue(out.shape == torch.Size([1, 576, 1, 1, 1]))
        features = md.forward_features(input_tensor)
        self.assertEqual(features.shape, (1, 576))
        score = md.head(features)
        self.assertEqual(score.shape, (1, 1000))
