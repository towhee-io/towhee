# Built on top of the original implementation at https://github.com/mesnico/Wiki-Image-Caption-Matching/blob/master/mcprop/model.py
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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

import torch
from torch import nn
from transformers import AutoModel


class TextExtractor(nn.Module):
    """
    Text extractor
    Args:
        text_model (str): text model
        finetune (bool): finetune
    """
    def __init__(self, text_model: str, finetune: bool):
        super().__init__()
        self.finetune = finetune
        self.text_model = AutoModel.from_pretrained(text_model)

    def forward(self, ids, mask):
        with torch.set_grad_enabled(self.finetune):
            out = self.text_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        out = torch.stack(out.hidden_states, dim=0)
        return out
