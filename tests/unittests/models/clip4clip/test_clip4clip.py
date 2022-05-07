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

from towhee.models.clip import get_configs
from towhee.models.clip4clip.clip4clip import CLIP4Clip


class TestCLIP4Clip(unittest.TestCase):
    """
    Test CLIP4Clip model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    configs = get_configs("clip_vit_b32")
    if "url" in configs:
        url = configs["url"]
        configs.pop("url")
    configs["context_length"] = 32
    model = CLIP4Clip(**configs)
    model.to(device)

    def test_forward(self):
        input_ids = torch.randint(low=0, high=2, size=(2, 1, 32))
        segment_ids = torch.randint(low=0, high=2, size=(2, 1, 32))
        input_mask = torch.randint(low=0, high=2, size=(2, 1, 32))
        video = torch.randn(2, 1, 12, 1, 3, 224, 224)
        video_mask = torch.randint(low=0, high=2, size=(2, 1, 12))
        loss = self.model(input_ids, segment_ids, input_mask, video, video_mask)
        self.assertTrue(loss.size() == torch.Size([]))


if __name__ == "__main__":
    unittest.main()
