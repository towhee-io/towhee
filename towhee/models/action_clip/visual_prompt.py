# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# Mengmeng Wang, Jiazheng Xing, Yong Liu
#
# Built on top of official implementation at https://github.com/sallymmx/ActionCLIP
#
# Modifications by Copyright 2021 Zilliz. All rights reserved.
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
from collections import OrderedDict

from towhee.models.utils.weight_init import trunc_normal_
from towhee.models.utils.init_vit_weights import init_vit_weights


class LayerNorm(nn.Module):
    """
    Construct a layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class QuickGELU(nn.Module):
    """
    QuickGELU
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    ResidualAttentionBlock
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, d_model * 4)),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TAggregate(nn.Module):
    """
    Aggregation module.

    Args:
        - clip_length (`int=0`):
            Dimension of CLIP features.
        - embed_dim (`int=2048`):
            Dimension used in Transformer encoder.
        - n_layers (`int=6`):
            Number of layers used in Transformer encoder.
    """
    def __init__(self, clip_length=0, embed_dim=2048, n_layers=6):
        super().__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(init_vit_weights)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]


class TemporalTransformer(nn.Module):
    """
    TemporalTransformer
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualPrompt(nn.Module):
    """
    VisualPrompt
    """
    def __init__(self, sim_head, clip_state_dict, num_frames):
        super().__init__()
        self.sim_header = sim_head
        self.num_frames = num_frames
        assert sim_head in ['meanP', 'LSTM', 'Transf', 'Conv_1D', 'Transf_cls']

        if self.sim_header == 'LSTM'\
                or self.sim_header == 'Transf'\
                or self.sim_header == 'Transf_cls'\
                or self.sim_header == 'Conv_1D':
            embed_dim = clip_state_dict['text_projection'].shape[1]

            context_length = clip_state_dict['positional_embedding'].shape[0]
            # vocab_size = clip_state_dict['token_embedding.weight'].shape[0]
            transformer_width = clip_state_dict['ln_final.weight'].shape[0]
            transformer_heads = transformer_width // 64

            # transformer_layers = len(
            #     set(k.split('.')[2] for k in clip_state_dict if k.startswith('transformer.resblocks')))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        if self.sim_header == 'Transf':
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
        if self.sim_header == 'LSTM':
            self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.apply(init_vit_weights)

        if self.sim_header == 'Transf_cls':
            self.transformer = TAggregate(clip_length=self.num_frames, embed_dim=embed_dim, n_layers=6)

        if self.sim_header == 'Conv_1D':
            self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
            weight = torch.zeros(embed_dim, 1, 3)
            weight[:embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4:, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

    def forward(self, x):
        _, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == 'meanP':
            pass
        elif self.sim_header == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == 'Transf':
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == 'LSTM':
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == 'Transf_cls':
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError(f'Unknown optimizer: {self.sim_header}')
        return x.mean(dim=1, keepdim=False)
