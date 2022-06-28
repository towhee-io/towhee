# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
from torch import nn
from towhee.models.lightning_dot.bi_encoder import BiEncoder


class MockUniterEncoder(nn.Module):
    """
    A Mock UniterEncoder
    """

    @classmethod
    def init_encoder(cls, config, checkpoint_path=None, project_dim=8):
        print(
            f"UniterEncoder init_encoder, config={config}, checkpoint_path={checkpoint_path}, project_dim={project_dim}")
        return MockUniterEncoder()

    def forward(self, input_ids, attention_mask, position_ids, img_feat, img_pos_feat, img_masks, gather_index):
        print(f"forward, input_ids={input_ids}, attention_mask={attention_mask}, position_ids={position_ids}"
              f"img_feat={img_feat}, img_pos_feat={img_pos_feat}, img_masks={img_masks}, gather_index={gather_index}")
        return torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1)


class MockBertEncoder(nn.Module):
    """
    A Mock BertEncoder
    """

    @classmethod
    def init_encoder(cls, config, checkpoint_path=None, project_dim=8):
        print(f"Encoder init_encoder, config={config}, checkpoint_path={checkpoint_path}, project_dim={project_dim}")
        return MockBertEncoder()

    def forward(self, input_ids, attention_mask, position_ids, img_feat, img_pos_feat, img_masks, gather_index):
        print(f"forward, input_ids={input_ids}, attention_mask={attention_mask}, position_ids={position_ids}"
              f"img_feat={img_feat}, img_pos_feat={img_pos_feat}, img_masks={img_masks}, gather_index={gather_index}")
        return torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1)


class MockArgs():
    def __init__(self, img_model_type="uniter-base", txt_model_type="bert-base"):
        self.img_model_type = img_model_type
        self.txt_model_type = txt_model_type
        self.img_model_config = "img_model_config"
        self.txt_model_config = "txt_model_config"
        self.img_checkpoint = "./"
        self.txt_checkpoint = "./"


class TestLightningDOT(unittest.TestCase):
    """
    Test LightningDOT model
    """
    args = MockArgs()
    model = BiEncoder(MockUniterEncoder(), MockBertEncoder(), args)

    def test_bi_encoder(self):
        batch = {
            "imgs":
                {
                    "input_ids": 1,
                    "attention_mask": 1,
                    "position_ids": 1,
                    "img_feat": 1,
                    "img_pos_feat": 1,
                    "img_masks": 1,
                    "gather_index": 1,
                },
            "txts":
                {
                    "input_ids": 1,
                    "attention_mask": 1,
                    "position_ids": 1,
                    "img_feat": 1,
                    "img_pos_feat": 1,
                    "img_masks": 1,
                    "gather_index": 1,
                },
            "caps":
                {
                    "input_ids": 1,
                    "attention_mask": 1,
                    "position_ids": 1,
                    "img_feat": 1,
                    "img_pos_feat": 1,
                    "img_masks": 1,
                    "gather_index": 1,
                }
        }
        out = self.model(batch)
        self.assertTrue(out[0].requires_grad is True)
        self.assertTrue(out[1].requires_grad is True)
        self.assertTrue(out[2].requires_grad is True)


if __name__ == "__main__":
    unittest.main()
