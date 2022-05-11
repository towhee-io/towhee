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
from towhee.models import clip


class TestActionClip(unittest.TestCase):
    """
    Test ActionClip model
    """
    video = torch.rand(1, 2, 3, 4, 4)
    text = ["archery", "dance"]
    clip_model = clip.create_model(
            embed_dim=512, image_resolution=4, vision_layers=12, vision_width=768, vision_patch_size=2,
            context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12)

    # Test video encoder
    def test_encode_video(self):
        vis_feats = action_clip.encode_video(self.video, self.clip_model, prompt="Transf")
        self.assertTrue(vis_feats.shape, (1, 512))

    # Test text encoder
    def test_text_prompt(self):
        classes, num_text_aug, text_dict= action_clip.encode_text(self.text, self.clip_model, return_long=True)
        self.assertTrue(classes.shape, (32, 77))
        self.assertTrue(text_dict[0].shape, (2, 77))
        self.assertEqual(num_text_aug, len(text_dict), 16)


if __name__ == "__main__":
    unittest.main()
