# Copyright 2022 Zilliz. All rights reserved.
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
from torch import nn

from towhee.models.coca.coca import CoCa

class CoCaTest(unittest.TestCase):
    def test_CoCa(self):
        """
        Test CoCa model.
        """
        model = nn.Sequential()
        coca = CoCa(
            dim = 4,
            img_encoder = model,
            image_dim = 32,
            num_tokens = 10,
            unimodal_depth = 6,
            multimodal_depth = 6,
            dim_head = 64,
            heads = 8,
            caption_loss_weight = 1.,
            contrastive_loss_weight = 1.,
        )

        text = torch.randint(0, 10, (4, 4))
        images = torch.randn(1, 2, 32)
        logits = coca(
            text = text,
            images = images
        )
        self.assertTrue(logits.shape == torch.Size([4,4,10]))
        text_embeds, image_embeds = coca(
            text = text,
            images = images,
            return_embeddings = True
        )
        self.assertTrue(text_embeds.shape == torch.Size([4,4]))
        self.assertTrue(image_embeds.shape == torch.Size([1,4]))
