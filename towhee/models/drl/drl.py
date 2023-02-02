# original code from https://github.com/foolwood/ddRL/blob/main/tvr/models/modeling.py
# modified by Zilliz

from collections import OrderedDict
from typing import Optional
from types import SimpleNamespace
import torch
import logging
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from towhee.models import clip

from towhee.models.drl.until_module import convert_weights
from towhee.models.drl.module_cross import CrossModel, Transformer as TransformerClip
from towhee.models.drl.until_module import LayerNorm, AllGather, AllGather2, CrossEn

allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)


class DRL(nn.Module):
    """
    This is a PyTorch implementation of the paper Disentangled Representation Learning for Text-Video Retrieval.
    Args:
        base_encoder (`str`):
            CLIP encoder backbone. default: `clip_vit_b32`
        agg_module (`str`):
            Feature aggregation module for video. default: `seqTransf`, choices=[`ndone`, `seqLSTM`, `seqTransf`]
        interaction (`str`):
            Interaction type for retrieval. default: `wti`.
        wti_arch (`int)`:
            Select a architecture for weight branch. default: 2.
        cdcr (`int`):
            Channel decorrelation regularization. default: 3.
        cdcr_alpha1 (`float`):
            Coefficient 1 for channel decorrelation regularization. default: 1.0.
        cdcr_alpha2 (`float`):
            Coefficient 2 for channel decorrelation regularization. default: 0.06.
        cdcr_lambda (`float`):
            Coefficient for channel decorrelation regularization. default: 0.001.
        cross_num_hidden_layers (`int`):
            Number of hidden layers for cross transformer interaction.
    """

    def __init__(self,
                 base_encoder: str = "clip_vit_b32",
                 agg_module: str = "seqTransf",
                 interaction: str = "wti",
                 wti_arch: int = 2,
                 cdcr: int = 3,
                 cdcr_alpha1: float = 1.0,
                 cdcr_alpha2: float = 0.06,
                 cdcr_lambda: float = 0.001,
                 cross_num_hidden_layers: Optional[int] = None,
                 backbone_pretrained: bool = False
                 ):
        super().__init__()
        self.base_encoder = base_encoder
        self.agg_module = agg_module
        self.interaction = interaction
        self.wti_arch = wti_arch
        self.cdcr = cdcr
        self.cdcr_alpha1 = cdcr_alpha1
        self.cdcr_alpha2 = cdcr_alpha2
        self.cdcr_lambda = cdcr_lambda

        self.agg_module = agg_module
        backbone = base_encoder

        self.clip = clip.create_model(model_name=backbone, pretrained=backbone_pretrained, jit=False, clip4clip=True)

        state_dict = self.clip.state_dict()
        context_length = state_dict["positional_embedding"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64

        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config
        if self.interaction == "xti":
            if cross_num_hidden_layers is not None:
                setattr(cross_config, "num_hidden_layers", cross_num_hidden_layers)

            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)
        elif self.interaction == "mlp":
            self.similarity_dense = nn.Sequential(nn.Linear(transformer_width * 2, transformer_width),
                                                  nn.ReLU(inplace=True), nn.Linear(transformer_width, 1))
        elif self.interaction == "wti":
            if self.wti_arch == 1:
                self.text_weight_fc = nn.Linear(transformer_width, 1)
                self.video_weight_fc = nn.Linear(transformer_width, 1)
            elif self.wti_arch == 2:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            elif self.wti_arch == 3:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,  # pylint: disable=invalid-name
                                                       layers=cross_config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)  # random init must before loading pretrain

        # ===> Initialization trick [HARdd COddE]
        new_state_dict = OrderedDict()
        if self.interaction == "xti":
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < cross_config.num_hidden_layers:
                            new_state_dict["cross." + key] = val.clone()
                            continue

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < cross_config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        # <=== End of initialization trick

    def forward(self, text_ids, text_mask, video, video_mask=None):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # bd x nd_v x 3 x H x W - >  (bd x nd_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        b, n_v, d, h, w = video.shape
        video = video.view(b * n_v, d, h, w)

        text_feat, video_feat = self.get_text_video_feat(text_ids, video, video_mask, shaped=True)

        if self.training:
            sim_matrix1, sim_matrix2, cdcr_loss = self.get_similarity_logits(text_feat, video_feat,
                                                                             text_mask, video_mask, shaped=True)
            sim_loss = (self.loss_fct(sim_matrix1) + self.loss_fct(sim_matrix2)) / 2.0
            loss = sim_loss + cdcr_loss * self.config.cdcr_lambda

            return loss
        else:
            return None

    def get_text_feat(self, text_ids, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])

        bs_pair = text_ids.size(0)
        text_feat = self.clip.encode_text(text_ids, clip4clip=True, return_hidden=True, device=text_ids.device)[1].float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))

        return text_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)

        bs_pair = video_mask.size(0)
        video_feat = self.clip.encode_image(video).float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))

        video_feat = self.aggvideo_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def get_text_video_feat(self, text_ids, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            # text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)

        text_feat = self.get_text_feat(text_ids, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def aggvideo_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "ndone":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training:
                self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # ndLdd -> Lnddd
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # Lnddd -> ndLdd
            video_feat = video_feat + video_feat_original
        return video_feat

    def dp_interaction(self, text_feat, video_feat, text_mask, video_mask):
        text_feat = self.get_text_sep_feat(text_feat, text_mask)  # bd x 1 x dd

        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat = allgather(text_feat, self.config)
            video_feat = allgather(video_feat, self.config)
            video_mask = allgather(video_mask, self.config)
            torch.distributed.barrier()  # force sync

        text_feat = text_feat.squeeze(1)  # bd x 1 x dd -> bd x dd
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # bd x dd

        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        video_feat = self.get_video_avg_feat(video_feat, video_mask)  # bd x nd_v x dd -> bd x dd
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.matmul(text_feat, video_feat.t())
        if self.training:
            logit_scale = self.clip.logit_scale.exp()  #
            retrieve_logits = logit_scale * retrieve_logits

            if self.config.cdcr != 0:
                z_a_norm = (text_feat - text_feat.mean(0)) / text_feat.std(0)  # bdxdd
                z_b_norm = (video_feat - video_feat.mean(0)) / video_feat.std(0)  # bdxdd

                # cross-correlation matrix
                bd, dd = z_a_norm.shape
                c = torch.einsum("bm,bn->mn", z_a_norm, z_b_norm) / bd  # ddxdd
                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(dd - 1, dd + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.config.cdcr_alpha1 + off_diag * self.config.cdcr_alpha2)

                return retrieve_logits, retrieve_logits.T, cdcr_loss
            else:
                return retrieve_logits, retrieve_logits.T, 0.0
        else:
            return retrieve_logits, retrieve_logits.T, 0.0

    def _get_cross_feat(self, text_feat, video_feat, text_mask, video_mask):
        concat_feats = torch.cat((text_feat, video_feat), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((text_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(text_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_feat = self.cross(concat_feats, concat_type, concat_mask,
                                               output_all_encoded_layers=True)
        cross_feat = cross_layers[-1]

        return cross_feat, pooled_feat, concat_mask

    def xti_interaction(self, text_feat, video_feat, text_mask, video_mask):

        text_feat = self.get_text_sep_feat(text_feat, text_mask)  # bd x 1 x dd

        b_text, s_text, d_text = text_feat.size()
        b_video, s_video, d_video = video_feat.size()
        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat_full = allgather2(text_feat, self.config)
            video_feat_full = allgather2(video_feat, self.config)
            video_mask_full = allgather2(video_mask, self.config)
            text_feat = text_feat_full[b_text * self.config.local_rank: b_text * (1 + self.config.local_rank)]
            video_feat = video_feat_full[b_video * self.config.local_rank: b_video * (1 + self.config.local_rank)]
            torch.distributed.barrier()  # force sync
        else:
            text_feat_full = text_feat
            video_feat_full = video_feat
            video_mask_full = video_mask

        b_text_full = text_feat_full.shape[0]
        b_video_full = video_feat_full.shape[0]

        text_mask = torch.ones(text_feat.size(0), 1).to(device=text_mask.device, dtype=text_mask.dtype)
        text_mask_full = torch.ones(text_feat_full.size(0), 1).to(device=text_mask.device, dtype=text_mask.dtype)

        # tV
        text_feat_1 = text_feat.unsqueeze(1).repeat(1, b_video_full, 1, 1)  # b_t x bd_v x n_t x d_t
        text_feat_1 = text_feat_1.view(-1, s_text, d_text)  # (b_t x bd_v) x n_t x d_t
        text_mask_1 = text_mask.unsqueeze(1).repeat(1, b_video_full, 1)  # b_t x bd_v x 1
        text_mask_1 = text_mask_1.view(-1, s_text)  # (b_t x bd_v) x 1

        video_feat_1 = video_feat_full.unsqueeze(0).repeat(b_text, 1, 1, 1)  # b_t x bd_v x n_v x d_t
        video_feat_1 = video_feat_1.view(-1, s_video, d_video)  # (b_t x bd_v) x n_v x d_v
        video_mask_1 = video_mask_full.unsqueeze(0).repeat(b_text, 1, 1)  # b_t x bd_v x n_v
        video_mask_1 = video_mask_1.view(-1, s_video)  # (b_t x bd_v) x n_v

        # vT
        text_feat_2 = text_feat_full.unsqueeze(1).repeat(1, b_video, 1, 1)  # bd_t x b_v x n_t x d_t
        text_feat_2 = text_feat_2.view(-1, s_text, d_text)  # (bd_t x b_v) x n_t x d_t
        text_mask_2 = text_mask_full.unsqueeze(1).repeat(1, b_video, 1)  # bd_t x b_v x 1
        text_mask_2 = text_mask_2.view(-1, s_text)  # (bd_t x b_v) x 1

        video_feat_2 = video_feat.unsqueeze(0).repeat(b_text_full, 1, 1, 1)  # bd_t x b_v x n_v x d_v
        video_feat_2 = video_feat_2.view(-1, s_video, d_video)  # (bd_t x b_v) x n_v x d_t
        video_mask_2 = video_mask.unsqueeze(0).repeat(b_text_full, 1, 1)  # bd_t x b_v x n_v
        video_mask_2 = video_mask_2.view(-1, s_video)  # (bd_t x b_v) x n_v

        _, pooled_feat, _ = \
            self._get_cross_feat(text_feat_1, video_feat_1, text_mask_1, video_mask_1)
        retrieve_logits_tv = self.similarity_dense(pooled_feat).squeeze(-1).view(b_text, b_video_full)
        _, pooled_feat, _ = \
            self._get_cross_feat(text_feat_2, video_feat_2, text_mask_2, video_mask_2)
        retrieve_logits_vt = self.similarity_dense(pooled_feat).squeeze(-1).view(b_text_full, b_video).T

        if self.training:
            logit_scale = self.clip.logit_scale.exp()  #
            retrieve_logits_tv = torch.roll(retrieve_logits_tv, -b_text * self.config.local_rank, -1)
            retrieve_logits_vt = torch.roll(retrieve_logits_vt, -b_video * self.config.local_rank, -1)
            retrieve_logits_tv = logit_scale * retrieve_logits_tv
            retrieve_logits_vt = logit_scale * retrieve_logits_vt

            return retrieve_logits_tv, retrieve_logits_vt, 0.0
        else:
            return retrieve_logits_tv, retrieve_logits_vt, 0.0

    def wti_interaction(self, text_feat, video_feat, text_mask, video_mask):
        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat = allgather(text_feat, self.config)
            video_feat = allgather(video_feat, self.config)
            text_mask = allgather(text_mask, self.config)
            video_mask = allgather(video_mask, self.config)
            torch.distributed.barrier()  # force sync

        if self.config.interaction == "wti":
            text_weight = self.text_weight_fc(text_feat).squeeze(2)  # bd x nd_t x dd -> bd x nd_t
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool),
                                     float("-inf"))  # pylint: disable=not-callable
            text_weight = torch.softmax(text_weight, dim=-1)  # bd x nd_t

            video_weight = self.video_weight_fc(video_feat).squeeze(2)  # bd x nd_v x dd -> bd x nd_v
            video_weight.masked_fill_(torch.tensor((1 - video_mask), dtype=torch.bool),
                                      float("-inf"))  # pylint: disable=not-callable
            video_weight = torch.softmax(video_weight, dim=-1)  # bd x nd_v

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum("atd,bvd->abtv", [text_feat, video_feat])
        retrieve_logits = torch.einsum("abtv,at->abtv", [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum("abtv,bv->abtv", [retrieve_logits, video_mask])
        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        # max for video token
        if self.config.interaction == "ti":  # token-wise interaction
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
            v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        elif self.config.interaction == "wti":  # weighted token-wise interaction
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            t2v_logits = torch.einsum("abt,at->ab", [t2v_logits, text_weight])

            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            v2t_logits = torch.einsum("abv,bv->ab", [v2t_logits, video_weight])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits

            if self.config.cdcr == 1:
                # simple random
                text_feat = text_feat[torch.arange(text_feat.shape[0]),
                            torch.randint_like(text_sum, 0, 10000) % text_sum, :]
                video_feat = video_feat[torch.arange(video_feat.shape[0]),
                             torch.randint_like(video_sum, 0, 10000) % video_sum, :]
                z_a_norm = (text_feat - text_feat.mean(0)) / text_feat.std(0)  # ndxnd_sxdd
                z_b_norm = (video_feat - video_feat.mean(0)) / video_feat.std(0)  # ndxnd_txdd

                # cross-correlation matrix
                bd, dd = z_a_norm.shape
                c = torch.einsum("ac,ad->cd", z_a_norm, z_b_norm) / bd  # ddxdd
                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(dd - 1, dd + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.config.cdcr_alpha1 + off_diag * self.config.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            elif self.config.cdcr == 2:
                # selecet max
                max_idx1 = max_idx1[torch.arange(max_idx1.shape[0]), torch.arange(max_idx1.shape[1])]
                max_idx2 = max_idx2[torch.arange(max_idx2.shape[0]), torch.arange(max_idx2.shape[1])]

                max_t_feat = text_feat[torch.arange(max_idx2.shape[0]).repeat_interleave(max_idx2.shape[1]),
                                       max_idx2.flatten()]
                max_v_feat = video_feat[torch.arange(max_idx1.shape[0]).repeat_interleave(max_idx1.shape[1]),
                                        max_idx1.flatten()]

                t_feat = text_feat.reshape(-1, text_feat.shape[-1])
                t_mask = text_mask.flatten().type(torch.bool)
                v_feat = video_feat.reshape(-1, text_feat.shape[-1])
                v_mask = video_mask.flatten().type(torch.bool)
                t_feat = t_feat[t_mask]
                v_feat = v_feat[v_mask]
                max_t_feat = max_t_feat[v_mask]
                max_v_feat = max_v_feat[t_mask]

                z_a_norm = (t_feat - t_feat.mean(0)) / t_feat.std(0)  # (bdxnd_t)xdd
                z_b_norm = (max_v_feat - max_v_feat.mean(0)) / max_v_feat.std(0)  # (bdxnd_t)xdd

                x_a_norm = (v_feat - v_feat.mean(0)) / v_feat.std(0)  # (bdxnd_v)xdd
                x_b_norm = (max_t_feat - max_t_feat.mean(0)) / max_t_feat.std(0)  # (bdxnd_v)xdd

                # cross-correlation matrix
                nd, dd = z_a_norm.shape
                c1 = torch.einsum("ac,ad->cd", z_a_norm, z_b_norm) / nd  # ddxdd
                nd, dd = x_a_norm.shape
                c2 = torch.einsum("ac,ad->cd", x_a_norm, x_b_norm) / nd  # ddxdd
                c = (c1 + c2) / 2.0
                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(dd - 1, dd + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.config.cdcr_alpha1 + off_diag * self.config.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            elif self.config.cdcr == 3:
                # selecet max
                max_idx1 = max_idx1[torch.arange(max_idx1.shape[0]), torch.arange(max_idx1.shape[1])]
                max_idx2 = max_idx2[torch.arange(max_idx2.shape[0]), torch.arange(max_idx2.shape[1])]

                max_t_feat = text_feat[torch.arange(max_idx2.shape[0]).repeat_interleave(max_idx2.shape[1]),
                                       max_idx2.flatten()].squeeze(1)
                max_v_feat = video_feat[torch.arange(max_idx1.shape[0]).repeat_interleave(max_idx1.shape[1]),
                                        max_idx1.flatten()].squeeze(1)

                t_feat = text_feat.reshape(-1, text_feat.shape[-1])
                t_mask = text_mask.flatten().type(torch.bool)
                v_feat = video_feat.reshape(-1, video_feat.shape[-1])
                v_mask = video_mask.flatten().type(torch.bool)
                t_feat = t_feat[t_mask]
                v_feat = v_feat[v_mask]
                max_t_feat = max_t_feat[v_mask]
                max_v_feat = max_v_feat[t_mask]
                text_weight = text_weight.flatten()[t_mask]
                video_weight = video_weight.flatten()[v_mask]

                z_a_norm = (t_feat - t_feat.mean(0)) / t_feat.std(0)  # (bdxnd_t)xdd
                z_b_norm = (max_v_feat - max_v_feat.mean(0)) / max_v_feat.std(0)  # (bdxnd_t)xdd

                x_a_norm = (v_feat - v_feat.mean(0)) / v_feat.std(0)  # (bdxnd_v)xdd
                x_b_norm = (max_t_feat - max_t_feat.mean(0)) / max_t_feat.std(0)  # (bdxnd_v)xdd

                # cross-correlation matrix
                nd, dd = z_a_norm.shape
                bd = text_feat.shape[0]
                c1 = torch.einsum("acd,a->cd", torch.einsum("ac,ad->acd", z_a_norm, z_b_norm),
                                  text_weight) / bd  # ddxdd
                c2 = torch.einsum("acd,a->cd", torch.einsum("ac,ad->acd", x_a_norm, x_b_norm),
                                  video_weight) / bd  # ddxdd
                c = (c1 + c2) / 2.0
                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = c.flatten()[1:].view(dd - 1, dd + 1)[:, :-1].pow_(2).sum()
                cdcr_loss = (on_diag * self.config.cdcr_alpha1 + off_diag * self.config.cdcr_alpha2)
                return retrieve_logits, retrieve_logits.T, cdcr_loss
            else:
                return retrieve_logits, retrieve_logits.T, 0.0
        else:
            return retrieve_logits, retrieve_logits.T, 0.0

    def get_similarity_logits(self, text_feat, video_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if self.interaction == "dp":
            t2v_logits, v2t_logits, cdcr_loss = self.dp_interaction(text_feat, video_feat, text_mask, video_mask)
        elif self.interaction == "xti":
            t2v_logits, v2t_logits, cdcr_loss = self.xti_interaction(text_feat, video_feat, text_mask, video_mask)
        elif self.interaction in ["ti", "wti"]:
            t2v_logits, v2t_logits, cdcr_loss = self.wti_interaction(text_feat, video_feat, text_mask, video_mask)
        else:
            raise NotImplementedError
        return t2v_logits, v2t_logits, cdcr_loss

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.ddataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def create_model(
        base_encoder: str = "clip_vit_b32",
        agg_module: str = "seqTransf",
        interaction: str = "wti",
        wti_arch: int = 2,
        cdcr: int = 3,
        cdcr_alpha1: float = 1.0,
        cdcr_alpha2: float = 0.06,
        cdcr_lambda: float = 0.001,
        cross_num_hidden_layers: int = None,
        pretrained: bool = False,
        weights_path: str = None,
        device: str = None
) -> DRL:
    """
    Build a DRL model.
    Args:
        base_encoder (`str`):
            Base_encoder in DRL model, `clip_vit_b32` or `clip_vit_b16`.
        agg_module (`str`):
            Feature aggregation module for video. default: `seqTransf`, choices=[`ndone`, `seqLSTM`, `seqTransf`]
        interaction (`str`):
            Interaction type for retrieval. default: `wti`.
        wti_arch (`int`):
            Select an architecture for weight branch. default: 2.
        cdcr (`int`):
            Channel decorrelation regularization. default: 3.
        cdcr_alpha1 (`float`):
            Coefficient 1 for channel decorrelation regularization. default: 1.0.
        cdcr_alpha2 (`float`):
            Coefficient 2 for channel decorrelation regularization. default: 0.06.
        cdcr_lambda (`float`):
            Coefficient for channel decorrelation regularization. default: 0.001.
        cross_num_hidden_layers (`int`):
            Number of hidden layers for cross transformer interaction.
        pretrained (`bool`):
            Whether model is pretrained, default if False.
        weights_path (`str`):
            Pretrained model local path, default if None.
        device (`str`):
            Model device. `cpu` or `cuda`.
    Returns:

    >>> from towhee.models import drl
    >>> model = drl.create_model("clip_vit_b32")
    >>> model.__class__.__name__
    'DRL'

    """
    model = DRL(base_encoder=base_encoder,
                agg_module=agg_module,
                interaction=interaction,
                wti_arch=wti_arch,
                cdcr=cdcr,
                cdcr_alpha1=cdcr_alpha1,
                cdcr_alpha2=cdcr_alpha2,
                cdcr_lambda=cdcr_lambda,
                cross_num_hidden_layers=cross_num_hidden_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if pretrained and weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata  # pylint: disable=protected-access

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(  # pylint: disable=protected-access
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():  # pylint: disable=protected-access
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

        if len(missing_keys) > 0:
            logger.info("Weights of %s not initialized from pretrained model: %s", model.__class__.__name__,
                        "\n   " + "\n   ".join(missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in %s: %s", model.__class__.__name__,
                        "\n   " + "\n   ".join(unexpected_keys))
        if len(error_msgs) > 0:
            logger.error("Weights from pretrained model cause errors in %s: %s", model.__class__.__name__,
                         "\n   " + "\n   ".join(error_msgs))

    if pretrained and weights_path is None:
        raise ValueError("weights_path is None")
    return model
