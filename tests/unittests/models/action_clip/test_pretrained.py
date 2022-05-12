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

from towhee.models import action_clip


class TestPretrained(unittest.TestCase):
    """
    Test ActionClip model with pretrained clip
    """
    video = torch.ones(1, 8, 3, 224, 224)
    text = ["a", "b"]
    model = action_clip.create_model(clip_model="clip_vit_b32", pretrained=True, jit=True)

    # Test video encoder
    def test_pretrained(self):
        vis_feats = self.model.encode_video(self.video)
        text_features = self.model.encode_text(self.text)
        num_augs = int(text_features.size(0) / len(self.text))
        similarity = action_clip.get_similarity(text_features, vis_feats, num_text_augs=num_augs)
        self.assertEqual(similarity.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
