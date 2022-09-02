# Built on top of the original implementation at https://github.com/papermsucode/mdmmt
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn.functional as F

from typing import Any, List, Dict
from collections import OrderedDict
from torch import nn
from towhee.models.mdmmt.bert_mmt import BertMMT


class ContextGating(nn.Module):
    """
    Context Gating Layer
    Args:
        dimension (int): dimension
        add_batch_norm (bool): add batch normalization
    """

    def __init__(self, dimension, add_batch_norm=True):
        super().__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedEmbeddingUnit(nn.Module):
    """
    Gated Embedding Unit
    Args:
        input_dimension (int): dimension of input
        output_dimension (int): dimension of output
        use_bn (bool): use batch normalization
        normalize (bool): normalization
    """

    def __init__(self, input_dimension, output_dimension, use_bn, normalize):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)
        self.normalize = normalize

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


class ReduceDim(nn.Module):
    """
    ReduceDim Layer
    """

    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        return x


def get_maxp(embd, mask):
    # (bs, ntok, embdim)
    # (bs, ntok) 1==token, 0==pad
    mask = mask.clone().to(dtype=torch.float32)
    all_pad_idxs = (mask == 0).all(dim=1)
    mask[mask == 0] = float("-inf")
    mask[mask == 1] = 0
    maxp = (embd + mask[..., None]).max(dim=1)[0]  # (bs, embdim)
    maxp[all_pad_idxs] = 0  # if there is not embeddings, use zeros
    return maxp


def pad(x, max_length):
    bs, n = x.shape
    if n < max_length:
        padding = torch.zeros(bs, max_length - n, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
    return x


class MMTTXT(nn.Module):
    """
    MMT TXT Model
    Args:
        txt_bert (nn.Module): Bert model using for text
        tokenizer (Any): Tokenizer for input text
        max_length (int): text max length
        modalities (List): modalities name list
        add_special_tokens (bool): whether use special tokens
        add_dot (bool): add dot in the end of text
        same_dim (int): same dim in the dimension of output of GatedEmbeddingUnit
        dout_prob (float): dropout prob
    """

    def __init__(self,
                 txt_bert: nn.Module,
                 tokenizer: Any,
                 max_length: int = 30,
                 modalities: List = None,
                 add_special_tokens: bool = True,
                 add_dot: bool = True,
                 same_dim: int = 512,
                 dout_prob: float = 0.1
                 ):
        super().__init__()

        self.orig_mmt_comaptible = int(os.environ.get("ORIG_MMT_COMPAT", 0))
        if self.orig_mmt_comaptible:
            print("ORIG_MMT_COMPAT")
        self.add_dot = add_dot
        self.add_special_tokens = add_special_tokens
        self.modalities = modalities
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.txt_bert = txt_bert
        text_dim = self.txt_bert.config.hidden_size

        self.text_gu = nn.ModuleDict()
        for mod in self.modalities:
            self.text_gu[mod] = GatedEmbeddingUnit(text_dim,
                                                   same_dim,
                                                   use_bn=True,
                                                   normalize=True)
        self.moe_fc_txt = nn.ModuleDict()
        self.moe_txt_dropout = nn.Dropout(dout_prob)
        for mod in self.modalities:
            self.moe_fc_txt[mod] = nn.Linear(text_dim, 1)

    @property
    def device(self):
        return next(self.parameters()).data.device

    def compute_weights_from_emb(self, embd):
        embd = self.moe_txt_dropout(embd)
        moe_weights = torch.cat([self.moe_fc_txt[mod](embd) for mod in self.modalities], dim=-1)
        moe_weights = F.softmax(moe_weights, dim=1)
        return moe_weights

    def forward(self, text_list):
        if self.add_dot:
            text_list1 = []
            for x in text_list:
                x = x.strip()
                if x[-1] not in (".", "?", "!"):
                    x = x + "."
                text_list1.append(x)
            text_list = text_list1
        device = self.device
        encoded_inputs = self.tokenizer(text_list,
                                        max_length=self.max_length,
                                        truncation=True,
                                        add_special_tokens=self.add_special_tokens,
                                        padding=True,
                                        return_tensors="pt")
        if self.orig_mmt_comaptible:
            encoded_inputs = {key: pad(value, self.max_length).to(device) for key, value in encoded_inputs.items()}
            encoded_inputs["head_mask"] = None
        else:
            encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
        x = self.txt_bert(**encoded_inputs)[0]  # (bs, max_tokens, hidden_size)
        # authors of MMT take token 0 and think that it is CLS
        # but they dont provide CLS token to input
        text = x[:, 0, :]
        text_embd = []
        for mod in self.modalities:
            layer = self.text_gu[mod]  # this layer containg F.normalize
            text_ = layer(text)  # (bs, d_model) this is unit-length embs
            if self.orig_mmt_comaptible:
                text_ = F.normalize(text_)
            text_ = text_.unsqueeze(1)  # (bs, 1, d_model)
            text_embd.append(text_)
        text_embd = torch.cat(text_embd, dim=1)  # (bs, nmods, d_model)
        if len(self.modalities) > 1:
            text_weights = self.compute_weights_from_emb(text)  # (bs, nmods)
            embd_wg = text_embd * text_weights[..., None]  # (bs, nmods, d_model)
        else:
            embd_wg = text_embd

        bs = embd_wg.size(0)
        text_embd = embd_wg.view(bs, -1)

        return text_embd


class MMTVID(nn.Module):
    """
    MMT VID model
    """

    def __init__(self,
                 expert_dims: Dict = None,
                 vid_bert_config: Any = None,
                 same_dim: int = 512,
                 hidden_size: int = 512,
                 ):
        super().__init__()
        self.modalities = list(expert_dims.keys())
        self.same_dim = same_dim
        self.expert_dims = expert_dims
        self.hidden_size = hidden_size  # self.vid_bert_params["hidden_size"]
        self.vid_bert_config = vid_bert_config
        self.vid_bert = BertMMT(vid_bert_config)

        self.video_dim_reduce = nn.ModuleDict()
        for mod in self.modalities:
            in_dim = expert_dims[mod]["dim"]
            self.video_dim_reduce[mod] = ReduceDim(in_dim, self.hidden_size)

        if same_dim != self.hidden_size:
            self.video_dim_reduce_out = nn.ModuleDict()
            for mod in self.modalities:
                self.video_dim_reduce_out[mod] = ReduceDim(self.hidden_size, same_dim)

    @property
    def device(self):
        return next(self.parameters()).data.device

    def forward(self,
                features,  # embs from pretrained models {modality: (bs, ntok, embdim)}
                features_t,  # timings {modality: (bs, ntok)} each value is (emb_t_start + emb_t_end) / 2
                features_ind,  # mask (modality: (bs, ntok))
                features_maxp=None,
                ):
        device = self.device
        experts_feats = dict(features)
        experts_feats_t = dict(features_t)
        experts_feats_ind = dict(features_ind)
        ind = {}  # 1 if there is at least one non-pad token in this modality
        for mod in self.modalities:
            ind[mod] = torch.max(experts_feats_ind[mod], 1)[0]

        for mod in self.modalities:
            layer = self.video_dim_reduce[mod]
            experts_feats[mod] = layer(experts_feats[mod])
        bs = next(iter(features.values())).size(0)
        ids_size = (bs,)
        input_ids_list = []
        token_type_ids_list = []  # Modality id
        # Position (0 = no position, 1 = unknown, >1 = valid position)
        position_ids_list = []
        features_list = []  # Semantics
        attention_mask_list = []  # Valid token or not

        modality_to_tok_map = OrderedDict()

        # 0=[CLS] 1=[SEP] 2=[AGG] 3=[MAXP] 4=[MNP] 5=[VLAD] 6=[FEA]
        # [CLS] token
        tok_id = 0
        input_ids_list.append(torch.full(ids_size, 0, dtype=torch.long))
        token_type_ids_list.append(torch.full(ids_size, 0, dtype=torch.long))
        position_ids_list.append(torch.full(ids_size, 0, dtype=torch.long).to(device))
        features_list.append(torch.full((bs, self.hidden_size), 0, dtype=torch.float).to(device))
        attention_mask_list.append(torch.full(ids_size, 1, dtype=torch.long).to(device))

        # Number of temporal tokens per modality
        max_expert_tokens = OrderedDict()
        for modality in self.modalities:
            max_expert_tokens[modality] = experts_feats[modality].size()[1]

        # Clamp the position encoding to [0, max_position_embedding - 1]
        max_pos = self.vid_bert_config.max_position_embeddings - 1
        for modality in self.modalities:
            experts_feats_t[modality].clamp_(min=0, max=max_pos)
            experts_feats_t[modality] = experts_feats_t[modality].long().to(device)

        for modality in self.modalities:
            token_type = self.expert_dims[modality]["idx"]
            tok_id += 1

            # add aggregation token
            modality_to_tok_map[modality] = tok_id
            input_ids_list.append(torch.full(ids_size, 2, dtype=torch.long))
            token_type_ids_list.append(torch.full(ids_size, token_type, dtype=torch.long))
            position_ids_list.append(torch.full(ids_size, 0, dtype=torch.long).to(device))
            layer = self.video_dim_reduce[modality]
            if features_maxp is not None:
                feat_maxp = features_maxp[modality]
            else:
                feat_maxp = get_maxp(features[modality], experts_feats_ind[
                    modality])
            features_list.append(layer(feat_maxp))
            attention_mask_list.append(ind[modality].to(dtype=torch.long).to(device))

            # add expert tokens
            for frame_id in range(max_expert_tokens[modality]):
                tok_id += 1
                position_ids_list.append(experts_feats_t[modality][:, frame_id])
                input_ids_list.append(torch.full(ids_size, 6, dtype=torch.long))
                token_type_ids_list.append(torch.full(ids_size, token_type, dtype=torch.long))
                features_list.append(experts_feats[modality][:, frame_id, :])
                attention_mask_list.append(experts_feats_ind[modality][:, frame_id].to(dtype=torch.long))

        features = torch.stack(features_list, dim=1).to(
            self.device)
        input_ids = torch.stack(input_ids_list, dim=1).to(self.device)
        token_type_ids = torch.stack(token_type_ids_list, dim=1).to(self.device)
        position_ids = torch.stack(position_ids_list, dim=1).to(self.device)
        attention_mask = torch.stack(attention_mask_list, dim=1).to(self.device)
        vid_bert_output = self.vid_bert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        features=features)
        last_layer = vid_bert_output[0]
        experts = []
        for modality in self.modalities:
            emb = last_layer[:, modality_to_tok_map[modality]]
            if self.same_dim != self.hidden_size:
                emb = self.video_dim_reduce_out[mod](emb)
            agg_tok_out = F.normalize(emb, dim=1)
            experts.append(agg_tok_out)  # (bs, embdim)
        experts = torch.cat(experts, dim=1)
        return experts
