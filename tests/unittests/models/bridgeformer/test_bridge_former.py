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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32
    patch_size = 16
    in_chans = 3
    text_len = 4
    num_frames = 4
    dummy_video = torch.randn(1, num_frames, in_chans, image_size, image_size).to(device)
    dummy_text = {}
    input_ids = torch.randint(1, 10, size=(1, text_len)).to(device)
    attention_mask = torch.ones(1, text_len, dtype=torch.int).to(device)
    dummy_text["input_ids"] = input_ids
    dummy_text["attention_mask"] = attention_mask
    dummy_ans = {"input_ids": input_ids, "attention_mask": attention_mask}
    dummy_question = {"input_ids": input_ids, "attention_mask": attention_mask}

    def test_base(self):
        """
        Test BridgeFormer without model_name=None & pretrained=False.
        """

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

    def test_clip_initialized_model(self):
        model = create_model(model_name="clip_initialized_model", pretrained=False,
                             image_resolution=self.image_size,
                             vision_patch_size=self.patch_size,
                             context_length=self.text_len
                             )
        text_embeddings = model.encode_text(self.input_ids)
        video_embeddings = model.encode_image(self.dummy_video)
        self.assertEqual(text_embeddings.shape, (1, 512))
        self.assertEqual(video_embeddings.shape, (1, 512))
        text_with_video_sim = sim_matrix(text_embeddings, video_embeddings)
        self.assertEqual(text_with_video_sim.shape, (1, 1))

    def test_bridge_former_training(self):
        model = create_model(pretrained=False, model_name="bridge_former_training",
                             img_size=self.image_size, patch_size=self.patch_size,
                             in_chans=self.in_chans,
                             projection_dim=256,
                             )
        text_cls_embeddings, answer_cls_embeddings, \
        bridge_cls_embeddings, video_cls_embeddings = model(text=self.dummy_text, answer_data=self.dummy_ans,
                                                            question_data=self.dummy_question, video=self.dummy_video)

        self.assertEqual(text_cls_embeddings.shape, (1, 256))
        self.assertEqual(answer_cls_embeddings.shape, (1, 256))
        self.assertEqual(bridge_cls_embeddings.shape, (1, 256))
        self.assertEqual(video_cls_embeddings.shape, (1, 256))


if __name__ == "__main__":
    unittest.main()
