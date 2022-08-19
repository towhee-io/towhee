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

from towhee.models.action_clip.visual_prompt import TAggregate


class TestPrompt(unittest.TestCase):
    """
    Test ActionClip prompts
    """
    # Test TAggregate
    def test_taggregate(self):
        x = torch.rand(1, 1, 8)
        layer = TAggregate(clip_length=0, embed_dim=8, n_layers=2)
        outs = layer(x)
        self.assertTrue(outs.shape == (1, 8))


if __name__ == "__main__":
    unittest.main()
