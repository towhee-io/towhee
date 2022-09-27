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
from transformers import AutoTokenizer

from towhee.models.mcprop.matching import Matching


class MatchingTest(unittest.TestCase):
    common_space_dim = 1024
    num_text_transformer_layers = 2
    img_feat_dim = 512
    txt_feat_dim = 768
    image_disabled = False
    aggregate_tokens_depth = None
    fusion_mode = 'weighted'
    text_model = 'xlm-roberta-base'
    finetune_text_model = False
    image_model = 'clip_vit_b32'
    finetune_image_model = False
    matcher = Matching(common_space_dim=common_space_dim, num_text_transformer_layers=num_text_transformer_layers,
                       img_feat_dim=img_feat_dim, txt_feat_dim=txt_feat_dim, image_disabled=image_disabled,
                       aggregate_tokens_depth=aggregate_tokens_depth, fusion_mode=fusion_mode, text_model=text_model,
                       finetune_text_model=finetune_text_model, image_model=image_model,
                       finetune_image_model=finetune_image_model)

    def test_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        max_length = 80
        caption = 'Hajnówka [SEP] loka monumento pri bizono'
        caption_inputs = tokenizer.encode_plus(
            caption,
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length'
        )
        caption_ids = caption_inputs['input_ids']
        caption_mask = caption_inputs['attention_mask']
        url = 'Zubr-Hajnowka'
        url_inputs = tokenizer.encode_plus(
            url,
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length'
        )
        url_ids = url_inputs['input_ids']
        url_mask = url_inputs['attention_mask']
        url_ids = torch.tensor([url_ids], dtype=torch.long)
        url_mask = torch.tensor([url_mask], dtype=torch.long)
        caption_ids = torch.tensor([caption_ids], dtype=torch.long)
        caption_mask = torch.tensor([caption_mask], dtype=torch.long)
        dummy_img = torch.rand(1, 3, 224, 224)
        query_feats, caption_feats, alphas = self.matcher.compute_embeddings(img=dummy_img, url=url_ids,
                                                                             url_mask=url_mask,
                                                                            caption=caption_ids,
                                                                             caption_mask=caption_mask)
        self.assertEqual(query_feats.shape, (1, 1024))
        self.assertEqual(caption_feats.shape, (1, 1024))
        self.assertEqual(alphas.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
