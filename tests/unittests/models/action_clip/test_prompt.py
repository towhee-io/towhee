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
from collections import OrderedDict

from towhee.models.action_clip.visual_prompt import TAggregate, VisualPrompt, TemporalTransformer


class TestPrompt(unittest.TestCase):
    """
    Test ActionClip prompts
    """
    # Test TAggregate
    def test_taggregate(self):
        x = torch.rand(1, 1, 8)
        layer = TAggregate(clip_length=0, embed_dim=8, n_layers=2)
        outs = layer(x)
        self.assertTrue(outs.shape == (1, 8))

    def test_visual_prompt_lstm(self):
        x = torch.rand(1, 4, 8)
        fake_clip_weights = OrderedDict()
        fake_clip_weights["text_projection"] = torch.rand(1, 8)
        fake_clip_weights["positional_embedding"] = torch.rand(4)
        fake_clip_weights["ln_final.weight"] = torch.rand(64, 1)

        model1 = VisualPrompt(sim_head="LSTM", clip_state_dict=fake_clip_weights, num_frames=4)
        out1 = model1(x)
        # print(type(model1.lstm_visual), out1.shape)
        self.assertTrue(isinstance(model1.lstm_visual, torch.nn.LSTM))
        self.assertTrue(out1.shape == (1, 8))

        model2 = VisualPrompt(sim_head="Transf", clip_state_dict=fake_clip_weights, num_frames=4)
        out2 = model2(x)
        # print(type(model2.transformer), out2.shape)
        self.assertTrue(isinstance(model2.transformer, TemporalTransformer))
        self.assertTrue(out2.shape == (1, 8))

        model3 = VisualPrompt(sim_head="Transf_cls", clip_state_dict=fake_clip_weights, num_frames=4)
        out3 = model3(x)
        # print(type(model3.transformer), out3.shape)
        self.assertTrue(isinstance(model3.transformer, TAggregate))
        self.assertTrue(out3.shape == (1, 8))

        model4 = VisualPrompt(sim_head="Conv_1D", clip_state_dict=fake_clip_weights, num_frames=4)
        out4 = model4(x)
        # print(type(model4.shift), out4.shape)
        self.assertTrue(isinstance(model4.shift, torch.nn.Conv1d))
        self.assertTrue(out4.shape == (1, 8))


if __name__ == "__main__":
    unittest.main()
