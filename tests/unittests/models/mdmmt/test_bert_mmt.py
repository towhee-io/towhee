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

from towhee.models.mdmmt.bert_mmt import BertMMT


class TestBertMMT(unittest.TestCase):
    """
    Test CLIP4Clip model
    """
    vid_bert_params = {
        "vocab_size_or_config_json_file": 10,
        "hidden_size": 512,
        "num_hidden_layers": 9,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.2,
        "attention_probs_dropout_prob": 0.2,
        "max_position_embeddings": 32,
        "type_vocab_size": 19,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "num_attention_heads": 8,
    }

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Struct(**vid_bert_params)
    model = BertMMT(config=config)

    def test_forward(self):
        input_ids = torch.randint(low=0, high=200, size=(8, 94))
        attention_mask = torch.randint(low=0, high=2, size=(8, 94))
        token_type_ids = torch.randint(low=0, high=2, size=(8, 94))
        position_ids = torch.randint(low=0, high=2, size=(8, 94))
        features = torch.rand(8, 94, 512)
        output = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            features=features)
        self.assertTrue(output[0].size() == (8, 94, 512))


if __name__ == "__main__":
    unittest.main()
