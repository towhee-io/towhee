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
from towhee.models import clip


class TestClip(unittest.TestCase):
    """
    Test CLIP model
    """
    def test_clip(self):
        img = torch.rand(1, 3, 4, 4)
        text = torch.randint(high=49408, size=(2, 77), dtype=torch.int32)
        model = clip.create_model(
            embed_dim=512, image_resolution=4, vision_layers=12, vision_width=768, vision_patch_size=2,
            context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12
            )
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)
        self.assertEqual(image_features.shape, (1, 512))
        self.assertEqual(text_features.shape, (2, 512))

        logits_per_img, logits_per_text = model(img, text)
        self.assertEqual(logits_per_img.shape, (1, 2))
        self.assertEqual(logits_per_text.shape, (2, 1))

    def test_models(self):
        img = torch.rand(1, 3, 224, 224)
        text = ["test text"]

        model = clip.create_model(
            model_name="clip_vit_b16",
            pretrained=False,
            )
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)
        self.assertTrue(image_features.shape, (1, 512))
        self.assertTrue(text_features.shape, (1, 512))
        logits_per_img, logits_per_text = model(img, text)
        self.assertEqual(logits_per_img.shape, (1, 1))
        self.assertEqual(logits_per_text.shape, (1, 1))

        model = clip.create_model(
            model_name="clip_vit_b32",
            pretrained=False
            )
        image_features = model.encode_image(img)
        text_features = model.encode_text(text, multilingual=False)
        self.assertEqual(image_features.shape, (1, 512))
        self.assertEqual(text_features.shape, (1, 512))
        logits_per_img, logits_per_text = model(img, text)
        self.assertEqual(logits_per_img.shape, (1, 1))
        self.assertEqual(logits_per_text.shape, (1, 1))

        model = clip.create_model(
            model_name="clip_resnet_r50",
            pretrained=False
            )
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)
        self.assertEqual(image_features.shape, (1, 1024))
        self.assertEqual(text_features.shape, (1, 1024))
        logits_per_img, logits_per_text = model(img, text)
        self.assertEqual(logits_per_img.shape, (1, 1))
        self.assertEqual(logits_per_text.shape, (1, 1))

        model = clip.create_model(
            model_name="clip_resnet_r101",
            pretrained=False
            )
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)
        self.assertTrue(image_features.shape, (1, 512))
        self.assertTrue(text_features.shape, (1, 512))
        logits_per_img, logits_per_text = model(img, text)
        self.assertEqual(logits_per_img.shape, (1, 1))
        self.assertEqual(logits_per_text.shape, (1, 1))

    # def test_tokenizer(self):
    #     text = ["a dog", "a cat"]
    #     text_tokens = clip.tokenize(text)
    #     self.assertEqual(text_tokens.shape, (2, 77))


if __name__ == "__main__":
    unittest.main()
