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
from towhee.models import timesformer


class TestTimesformer(unittest.TestCase):
    """
    Test TimeSformer model
    """
    # timesformer_k400_8x224 with attention_type='divided_space_time',
    def test_timesformer_8(self):
        video = torch.randn(1, 3, 4, 4, 4)
        model = timesformer.create_model(
            model_name='timesformer_k400_8x224',
            pretrained=False,
            img_size=4,
            patch_size=2
            )
        pred = model(video)
        feats = model.forward_features(video)
        self.assertTrue(pred.shape == (1, 400))
        self.assertTrue(feats.shape == (1, 768))

    def test_timesformer_96(self):
        video = torch.randn(1, 3, 4, 4, 4)
        model = timesformer.create_model(
            model_name='timesformer_k400_96x224',
            pretrained=False,
            img_size=4,
            patch_size=2
            )
        pred = model(video)
        feats = model.forward_features(video)
        self.assertTrue(pred.shape == (1, 400))
        self.assertTrue(feats.shape == (1, 768))

    def test_other_types(self):
        video = torch.randn(1, 3, 2, 4, 4)
        for attention_type in ['space_only', 'joint_space_time']:
            model = timesformer.create_model(
                pretrained=False,
                img_size=4, num_classes=400, num_frames=2, patch_size=2, attention_type=attention_type)
            pred = model(video)
            feats = model.forward_features(video)
            self.assertTrue(pred.shape == (1, 400))
            self.assertTrue(feats.shape == (1, 768))


if __name__ == '__main__':
    unittest.main()
