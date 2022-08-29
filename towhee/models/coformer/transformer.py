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
#
# Original code from https://github.com/jhcho99/CoFormer.
#
# Modified by Zilliz.

"""
Transformer Architectures in CoFormer
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    Transformer class.
    Args:
        d_model (int): model dimension
        nhead (int): number of heads
        num_glance_enc_layers (int): number of glance encoder layers
        num_gaze_s1_dec_layers (int): number of gaze s1 decoder layers
        num_gaze_s2_dec_layers (int): number of gaze s2 decoder layers
        dim_feedforward (int): dimension of feedforward
        dropout (int): drop out probability
        activation (str): activation
    """
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_glance_enc_layers=3,
                 num_gaze_s1_dec_layers=3,
                 num_gaze_s1_enc_layers=3,
                 num_gaze_s2_dec_layers=3,
                 dim_feedforward=2048,
                 dropout=0.15,
                 activation="relu"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_verb_classes = 504

        # Glacne Transformer
        glance_enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.glance_enc = TransformerEncoder(glance_enc_layer, num_glance_enc_layers)

        # Gaze-Step1 Transformer
        gaze_s1_dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.gaze_s1_dec = TransformerDecoder(gaze_s1_dec_layer, num_gaze_s1_dec_layers)
        gaze_s1_enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.gaze_s1_enc = TransformerEncoder(gaze_s1_enc_layer, num_gaze_s1_enc_layers)

        # Gaze-Step2 Transformer
        gaze_s2_dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.gaze_s2_dec = TransformerDecoder(gaze_s2_dec_layer, num_gaze_s2_dec_layers)

        # classifer (for verb prediction)
        self.verb_classifier = nn.Sequential(nn.Linear(d_model*2, d_model*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(d_model*2, self.num_verb_classes))

        # layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model*2)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src,
                mask,
                il_token_embed,
                rl_token_embed,
                verb_token_embed,
                role_token_embed,
                pos_embed,
                vidx_ridx,
                targets=None,
                inference=False
               ):
        device = il_token_embed.device
        # flatten NxCxHxW to HWxNxC
        bs, _, h, w = src.shape
        flattend_src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        # Glance Transformer
        ## Encoder
        il_token = il_token_embed.unsqueeze(1).repeat(1, bs, 1)
        glance_enc_zero_mask = torch.zeros((bs, 1), dtype=torch.bool, device=device)
        mem_mask = torch.cat([glance_enc_zero_mask, mask], dim=1)
        il_token_flattend_src = torch.cat([il_token, flattend_src], dim=0)
        glance_enc_memory = self.glance_enc(il_token_flattend_src, src_key_padding_mask=mem_mask, pos=pos_embed, num_zeros=1)
        il_token_feature, aggregated_src = glance_enc_memory.split([1, h*w], dim=0)

        # Gaze-Step1 Transformer
        ## Decoder
        all_role_tokens = role_token_embed.unsqueeze(1).repeat(1, bs, 1)
        role_tgt = torch.zeros_like(all_role_tokens)
        extracted_rhs = self.gaze_s1_dec(all_role_tokens, self.ln1(flattend_src), memory_key_padding_mask=mask, pos=pos_embed, query_pos=role_tgt)
        extracted_rhs = extracted_rhs.transpose(1, 2)
        ## Encoder
        num_all_roles = 190
        rl_token = rl_token_embed.unsqueeze(1).repeat(1, bs, 1)
        gaze_s1_enc_zero_mask = torch.zeros((bs, (1 + num_all_roles)), dtype=torch.bool, device=device)
        rl_token_extracted_rhs = torch.cat([rl_token, extracted_rhs.view(num_all_roles, 1, -1)], dim=0)
        gaze_s1_enc_memory = self.gaze_s1_enc(rl_token_extracted_rhs,
                                              src_key_padding_mask=gaze_s1_enc_zero_mask,
                                              pos=None,
                                              num_zeros=(1+num_all_roles)
                                             )
        rl_token_feature, aggregated_rhs = gaze_s1_enc_memory.split([1, num_all_roles], dim=0)

        # Verb Prediction
        il_token_feature = il_token_feature.view(bs, -1)
        rl_token_feature = rl_token_feature.view(bs, -1)
        vhs = torch.cat([il_token_feature, rl_token_feature], dim=-1)
        vhs = self.ln2(vhs)
        verb_pred = self.verb_classifier(vhs).view(bs, self.num_verb_classes)

        # Gaze-Step2 Transformer
        ## Deocder
        ### At training time, we assume that the ground-truth verb is given.
        ##### Please see the evaluation details in [Grounded Situation Recognition] task.
        ##### There are three evaluation settings: Top-1 Predicted Verb, Top-5 Predicted Verbs and Ground-Truth Verb.
        ##### If top-1 predicted verb is incorrect, then grounded noun predictions in Top-1 Predicted Verb setting are considered incorrect.
        ##### If the ground-truth verb is not included in top-5 predicted verbs,
        # then grounded noun predictions in Top-5 Predicted Verbs setting are considered incorrect.
        ##### In Ground-Truth Verb setting, we only consider grounded noun predictions.
        ### At inference time, we use the predicted verb.
        #### For frame-role queries, we select the verb token embedding corresponding to the predicted verb.
        #### For frame-role queries, we select the role token embeddings corresponding to the roles associated with the predicted verb.
        if not inference:
            selected_verb_token = verb_token_embed[targets["verbs"]].view(1, -1)
            selected_roles = targets["roles"]
        else:
            top1_verb = torch.topk(verb_pred, k=1, dim=1)[1].item()
            selected_verb_token = verb_token_embed[top1_verb].view(1, -1)
            selected_roles = vidx_ridx[top1_verb]
        selected_role_tokens = role_token_embed[selected_roles]
        frame_role_queries = selected_role_tokens + selected_verb_token
        frame_role_queries = frame_role_queries.unsqueeze(1).repeat(1, bs, 1)
        role_tgt = torch.zeros_like(frame_role_queries)
        final_rhs = self.gaze_s2_dec(frame_role_queries, self.ln3(aggregated_src), memory_key_padding_mask=mask, pos=pos_embed, query_pos=role_tgt)
        final_rhs = self.ln4(final_rhs)
        final_rhs = final_rhs.transpose(1,2)

        return verb_pred, extracted_rhs, aggregated_rhs, final_rhs, selected_roles


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder class.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, num_zeros=num_zeros)

        return output


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder class.
    Args:
        decoder_layer (nn.Module): dencoder module
        num_layers (int): number of layers
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer class.
    Args:
        d_model (int): dimension of model
        nhead (int): number of head
        dim_feedforward (int): dimension of feedforward
        dropout (float): dropout
        activation (str): activation
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor], num_zeros=None):
        if num_zeros is not None:
            return tensor if pos is None else torch.cat([tensor[:num_zeros], (tensor[num_zeros:] + pos)], dim=0)
        else:
            return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos, num_zeros=num_zeros)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """
    TransformerDecoderLayer class.
    Args:
        d_model (int): dimension of model
        nhead (int): number of head
        dim_feedforward (int): dimension of feedforward
        dropout (float): dropout
        activation (str): activation
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def build_transformer(
                        d_model=512,
                        dropout=0.15,
                        nhead=8,
                        num_glance_enc_layers=3,
                        num_gaze_s1_dec_layers=3,
                        num_gaze_s1_enc_layers=3,
                        num_gaze_s2_dec_layers=3,
                        dim_feedforward=2048
                     ):
    return Transformer(
                        d_model=d_model,
                        dropout=dropout,
                        nhead=nhead,
                        num_glance_enc_layers=num_glance_enc_layers,
                        num_gaze_s1_dec_layers=num_gaze_s1_dec_layers,
                        num_gaze_s1_enc_layers=num_gaze_s1_enc_layers,
                        num_gaze_s2_dec_layers=num_gaze_s2_dec_layers,
                        dim_feedforward=dim_feedforward
                      )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
