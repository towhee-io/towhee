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
from torch import nn
from towhee.models.mdmmt.mmt import MMTVID, MMTTXT


class TestMMT(unittest.TestCase):
    """
    Test MMT model
    """

    def test_mmt_vid(self):
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
        expert_dims = {
            "CLIP": {"dim": 512, "idx": 2, "max_tok": 30},
            "tf_vggish": {"dim": 128, "idx": 3, "max_tok": 30},
        }
        features = {
            "CLIP": torch.rand(8, 30, 512),
            "tf_vggish": torch.rand(8, 30, 128)
        }
        features_t = {
            "CLIP": torch.rand(8, 30),
            "tf_vggish": torch.rand(8, 30)
        }
        features_ind = {
            "CLIP": torch.randint(low=0, high=2, size=(8, 30)),
            "tf_vggish": torch.randint(low=0, high=2, size=(8, 30))
        }

        mmtvid_model = MMTVID(
            expert_dims=expert_dims,
            same_dim=512,
            hidden_size=512,
            vid_bert_config=config
        )
        output = mmtvid_model(
            features=features,
            features_t=features_t,
            features_ind=features_ind,
            features_maxp=None,
        )
        self.assertTrue(output.shape == (8, 1024))

    def test_mmt_txt(self):
        batch_size = 8

        class MockBert(nn.Module):
            """
            Mock BERT model in transformers
            """

            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

                class Struct:
                    def __init__(self, **entries):
                        self.__dict__.update(entries)

                config = Struct(**{"hidden_size": hidden_size})
                self.config = config

            def forward(self, **kwargs):
                x = kwargs["input_ids"]
                return torch.rand(x.shape[0], x.shape[1], self.hidden_size), torch.rand(x.shape[0], self.hidden_size)

        class MockTokenizer:
            """
            Mock tokenizer in transformers
            """

            def __init__(self, vocab_size):
                self.vocab_size = vocab_size

            def __call__(self, text, max_length=30, truncation=True, add_special_tokens=None, padding=True,
                         return_tensors="pt"):
                print("tokenize text:", text)

                return {"input_ids": torch.randint(low=0, high=self.vocab_size, size=(batch_size, max_length,))}

        mmttxt_model = MMTTXT(
            txt_bert=MockBert(hidden_size=768),
            tokenizer=MockTokenizer(vocab_size=100),
            max_length=30,
            modalities=["CLIP", "tf_vggish"],
            add_special_tokens=True,
            add_dot=True,
            same_dim=512,
            dout_prob=0.2,
        )
        output = mmttxt_model([
            "hello world",
            "how are you"
        ])
        self.assertTrue(output.shape == (batch_size, 1024))


if __name__ == "__main__":
    unittest.main()
