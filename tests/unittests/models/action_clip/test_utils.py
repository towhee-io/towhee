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


class TestUtils(unittest.TestCase):
    """
    Test ActionClip utils
    """
    # Test Remap Weights of Visual Prompt
    def test_visual_prompt_weights(self):
        fake_weights = {"module.layer1": 1}
        new_weights = action_clip.map_state_dict(fake_weights)
        self.assertTrue("layer1" in new_weights)

    # Test similarity
    def test_similarity(self):
        text_features = torch.ones(6, 10)
        fusion_features = torch.ones(1, 10)
        similarity = action_clip.get_similarity(text_features, fusion_features, num_text_augs=3, norm=True)
        self.assertTrue(similarity[0].tolist() == [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
