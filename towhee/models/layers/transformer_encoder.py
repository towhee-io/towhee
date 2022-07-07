# Inspired by https://github.com/SvipRepetitionCounting/TransRAC/blob/main/models/TransRAC.py
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
from towhee.models.layers.position_encoding import PositionEmbedding


class TransformerEncoder(nn.Module):
    """
    standard transformer encoder

    Example:
        >>> import torch
        >>> from towhee.models.layers.transformer_encoder import TransformerEncoder
        >>>
        >>> dummpy_x = torch.rand(2, 4)
        >>> mode = TransformerEncoder(d_model=4, n_head=1, dim_ff=1, dropout=0.0, num_layers=1, num_frames=2)
        >>> out = mode(dummy_x)
        >>> print(out.shape)
        torch.Size([2, 2, 4])
    """

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super().__init__()
        self.pos_embed = PositionEmbedding(d_model=d_model, max_len=num_frames, dropout=dropout)

        encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                             nhead=n_head,
                                             dim_feedforward=dim_ff,
                                             dropout=dropout,
                                             activation='relu')
        norm_layer = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers, norm_layer)

    def forward(self, src):
        src = self.pos_embed(src)
        e_op = self.transformer_encoder(src)
        return e_op
