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
from towhee.models.frozen_in_time import FrozenInTime


class BridgeFormerTest(unittest.TestCase):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # (batch，frames，channels，height， width)
    image_size = 28
    patch_size = 16
    in_chans = 3
    dummy_video = torch.randn(1, 4, in_chans, image_size, image_size)
    dummy_text = {}
    input_ids = torch.randint(1, 10, size=(1, 4))
    attention_mask = torch.ones(1, 4, dtype=torch.int)
    dummy_text['input_ids'] = input_ids
    dummy_text['attention_mask'] = attention_mask

    def test_without_all_pretrained(self):
        '''
        do not use any pretrained model
        Returns:None
        '''

        model = FrozenInTime(img_size=self.image_size, patch_size=self.patch_size,
                             in_chans=self.in_chans,
                             attention_style='bridge_former',
                             is_pretrained=False,
                             projection_dim=256,
                             video_is_load_pretrained=False,
                             video_model_type='SpaceTimeTransformer',
                             text_is_load_pretrained=False,
                             device=self.device)
        text_embeddings, video_embeddings = model(text=self.dummy_text, video=self.dummy_video, return_embeds=True)
        self.assertEqual(text_embeddings.shape, (1, 256))
        self.assertEqual(video_embeddings.shape, (1, 256))
        text_with_video_sim = model(text=self.dummy_text, video=self.dummy_video, return_embeds=False)
        self.assertEqual(text_with_video_sim.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
