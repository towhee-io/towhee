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
from types import SimpleNamespace
from towhee.models.drl.module_cross import CrossModel
from towhee.models.drl.drl import DRL


class TestDRL(unittest.TestCase):
    """
    Test DRL model
    """

    def test_init(self):
        model = DRL()
        state_dict = model.clip.state_dict()
        # 77
        context_length = state_dict["positional_embedding"].shape[0]
        self.assertTrue(context_length == 77)

    def test_crossmodel(self):
        cross_config = SimpleNamespace(**{
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
        })
        cross = CrossModel(cross_config)
        width = cross.transformer.width
        self.assertTrue(width == 512)
        tes = cross.pooler.state_dict()["ln_pool.bias"].shape[0]  # pylint: disable=unsubscriptable-object
        self.assertTrue(tes == 512)


if __name__ == "__main__":
    unittest.main()
