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

from torch import nn


class TransformerPooling(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, num_layers=2):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4,
                                                         dim_feedforward=input_dim,
                                                         dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                           num_layers=num_layers)
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, input, mask):
        mask_bool = mask.clone()
        mask_bool = mask_bool.bool()
        mask_bool = ~mask_bool
        input = input.permute(1, 0, 2)
        output = self.transformer_encoder(input, src_key_padding_mask=mask_bool)
        output = output[0]  # take the CLS
        if self.proj is not None:
            output = self.proj(output)
        return output
