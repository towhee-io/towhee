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
from towhee.models.bridgeformer import create_model
from towhee.models.frozen_in_time.frozen_utils import sim_matrix


class BridgeFormerTest(unittest.TestCase):
    # (batch，frames，channels，height， width)
    image_size = 32
    patch_size = 16
    in_chans = 3
    text_len = 4
    dummy_video = torch.randn(1, 4, in_chans, image_size, image_size)
    dummy_text = {}
    input_ids = torch.randint(1, 10, size=(1, text_len))
    attention_mask = torch.ones(1, text_len, dtype=torch.int)
    dummy_text["input_ids"] = input_ids
    dummy_text["attention_mask"] = attention_mask

    def test_without_all_pretrained(self):
        '''
        do not use any pretrained model
        Returns:None
        '''

        model = create_model(pretrained=False,
                             img_size=self.image_size, patch_size=self.patch_size,
                             in_chans=self.in_chans,
                             projection_dim=256,
                             )
        text_embeddings, video_embeddings = model(text=self.dummy_text, video=self.dummy_video, return_embeds=True)
        self.assertEqual(text_embeddings.shape, (1, 256))
        self.assertEqual(video_embeddings.shape, (1, 256))
        text_with_video_sim = sim_matrix(text_embeddings, video_embeddings)
        self.assertEqual(text_with_video_sim.shape, (1, 1))

    def test_without_all_pretrained_with_clip_initialized_model(self):
        model = create_model(pretrained=False, model_name="clip_initialized_model", embed_dim=512,
                             image_resolution=self.image_size, vision_layers=12, vision_width=768,
                             vision_patch_size=self.patch_size,
                             context_length=self.text_len, vocab_size=49408, transformer_width=512, transformer_heads=8,
                             transformer_layers=12,
                             )
        text_embeddings = model.encode_text(self.input_ids)
        video_embeddings = model.encode_image(self.dummy_video)
        self.assertEqual(text_embeddings.shape, (1, 512))
        self.assertEqual(video_embeddings.shape, (1, 512))
        text_with_video_sim = sim_matrix(text_embeddings, video_embeddings)
        self.assertEqual(text_with_video_sim.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
