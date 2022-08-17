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
import numpy as np

from towhee.models.clip import SimpleTokenizer
from towhee.models.clip4clip import convert_tokens_to_id
from towhee.models.clip4clip import create_model
from towhee.models import clip4clip


class TestCLIP4Clip(unittest.TestCase):
    """
    Test CLIP4Clip model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name="clip_vit_b32", context_length=32, pretrained=False, device=device)

    def test_forward(self):
        input_ids = torch.randint(low=0, high=2, size=(2, 1, 32))
        video = torch.randn(2, 1, 12, 1, 3, 224, 224).to(self.device)
        video_mask = torch.randint(low=0, high=2, size=(2, 1, 12)).to(self.device)
        loss = self.model(input_ids, video, video_mask)
        self.assertTrue(loss.size() == torch.Size([]))

    def test_token(self):
        text = "hello world"
        text_tokens = clip4clip.tokenize(text)
        self.assertEqual(text_tokens, ["hello</w>", "world</w>"])

    def test_convert_tokens_to_id(self):
        tokenizer = SimpleTokenizer()
        text = "kids feeding and playing with the horse"
        res = convert_tokens_to_id(tokenizer, text, max_words=32)
        self.assertEqual(res.all(), np.array([[49406, 1911, 9879, 537, 1629, 593, 518, 4558, 49407] + [0] * 23]).all())


if __name__ == "__main__":
    unittest.main()
