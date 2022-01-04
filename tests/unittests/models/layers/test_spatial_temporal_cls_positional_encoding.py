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

from towhee.models.layers.spatial_temporal_cls_positional_encoding import SpatialTemporalClsPositionalEncoding


class SpaTempoClsPosEncodeTest(unittest.TestCase):
    def test_spatial_temporal_cls_positional_encoding(self):
        # Test with cls token.
        batch_dim = 4
        dim = 16
        video_shape = (1, 2, 4)
        video_sum = video_shape[0] * video_shape[1] * video_shape[2]
        has_cls = True
        model = SpatialTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=video_shape,
            has_cls=has_cls,
        )
        fake_input = torch.rand(batch_dim, video_sum, dim)
        output = model(fake_input)
        output_gt_shape = (batch_dim, video_sum + 1, dim)
        self.assertTrue(tuple(output.shape) == output_gt_shape)
