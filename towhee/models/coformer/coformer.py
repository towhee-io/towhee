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

import torch
import torch.nn.functional as F
from torch import nn
from towhee.models.coformer.utils import nested_tensor_from_tensor_list
from towhee.models.coformer.backbone import build_backbone
from towhee.models.coformer.transformer import build_transformer
from towhee.models.coformer.config import _C


class CoFormer(nn.Module):
    """
    CoFormer model for Grounded Situation Recognition
    Args:
        backbone (`nn.Module`):
            Torch module of the backbone to be used. See backbone.py.
        transformer (`nn.Module`):
            Torch module of the transformer architecture. See transformer.py.
        num_noun_classes (`int`):
            The number of noun classes.
        vidx_ridx (`dict`):
            Verb index to role index.
    """

    def __init__(self, backbone, transformer, num_noun_classes, vidx_ridx):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_noun_classes = num_noun_classes
        self.vidx_ridx = vidx_ridx
        self.num_role_tokens = 190
        self.num_verb_tokens = 504

        # hidden dimension for tokens and image features
        hidden_dim = transformer.d_model

        # token embeddings
        self.role_token_embed = nn.Embedding(self.num_role_tokens, hidden_dim)
        self.verb_token_embed = nn.Embedding(self.num_verb_tokens, hidden_dim)
        self.il_token_embed = nn.Embedding(1, hidden_dim)
        self.rl_token_embed = nn.Embedding(1, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # classifiers & predictors (for grounded noun prediction)
        self.noun_1_classifier = nn.Linear(hidden_dim, self.num_noun_classes)
        self.noun_2_classifier = nn.Linear(hidden_dim, self.num_noun_classes)
        self.noun_3_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                               nn.ReLU(),
                                               nn.Dropout(0.3),
                                               nn.Linear(hidden_dim * 2, self.num_noun_classes))
        self.bbox_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim * 2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.2),
                                                 nn.Linear(hidden_dim * 2, 1))

        # layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, samples, targets=None, inference=False):
        """
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb', 'pred_noun', 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        max_num_roles = 6
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        batch_size = src.shape[0]
        batch_verb, batch_noun_1, batch_noun_2, batch_noun_3, batch_bbox, batch_bbox_conf = [], [], [], [], [], []
        # model prediction
        for i in range(batch_size):
            if not inference:
                outs = self.transformer(self.input_proj(src[i:i + 1]),
                                        mask[i:i + 1], self.il_token_embed.weight, self.rl_token_embed.weight,
                                        self.verb_token_embed.weight, self.role_token_embed.weight,
                                        pos[-1][i:i + 1], self.vidx_ridx, targets=targets[i], inference=inference)
            else:
                outs = self.transformer(self.input_proj(src[i:i + 1]),
                                        mask[i:i + 1], self.il_token_embed.weight, self.rl_token_embed.weight,
                                        self.verb_token_embed.weight, self.role_token_embed.weight,
                                        pos[-1][i:i + 1], self.vidx_ridx, inference=inference)

            # output features & predictions
            verb_pred, extracted_rhs, aggregated_rhs, final_rhs, selected_roles = outs[0], outs[1], outs[2], outs[3], \
                                                                                  outs[4]
            num_selected_roles = len(selected_roles)
            # auxiliary classifiers
            if not inference:
                extracted_rhs = self.ln1(extracted_rhs[:, :, selected_roles, :])
                noun_1_pred = self.noun_1_classifier(extracted_rhs)
                noun_1_pred = F.pad(noun_1_pred,
                                    (0, 0, 0, max_num_roles - num_selected_roles),
                                    mode='constant',
                                    value=0,
                                    )[-1].view(1, max_num_roles, self.num_noun_classes)
                aggregated_rhs = self.ln2(
                    aggregated_rhs[selected_roles].permute(1, 0, 2).view(1, 1, num_selected_roles, -1))
                noun_2_pred = self.noun_2_classifier(aggregated_rhs)
                noun_2_pred = F.pad(noun_2_pred,
                                    (0, 0, 0, max_num_roles - num_selected_roles),
                                    mode='constant',
                                    value=0,
                                    )[-1].view(1, max_num_roles, self.num_noun_classes)
            else:
                noun_1_pred = None
                noun_2_pred = None
            noun_3_pred = self.noun_3_classifier(final_rhs)
            noun_3_pred = F.pad(noun_3_pred,
                                (0, 0, 0, max_num_roles - num_selected_roles),
                                mode='constant',
                                value=0,
                                )[-1].view(1, max_num_roles, self.num_noun_classes)
            bbox_pred = self.bbox_predictor(final_rhs).sigmoid()
            bbox_pred = F.pad(bbox_pred, (0, 0, 0, max_num_roles - num_selected_roles), mode='constant', value=0)[
                -1].view(1, max_num_roles, 4)
            bbox_conf_pred = self.bbox_conf_predictor(final_rhs)
            bbox_conf_pred = \
                F.pad(bbox_conf_pred, (0, 0, 0, max_num_roles - num_selected_roles), mode='constant', value=0)[-1].view(
                    1,
                    max_num_roles,
                    1)

            batch_verb.append(verb_pred)
            batch_noun_1.append(noun_1_pred)
            batch_noun_2.append(noun_2_pred)
            batch_noun_3.append(noun_3_pred)
            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)

        # outputs
        out = {}
        out['pred_verb'] = torch.cat(batch_verb, dim=0)
        if not inference:
            out['pred_noun_1'] = torch.cat(batch_noun_1, dim=0)
            out['pred_noun_2'] = torch.cat(batch_noun_2, dim=0)
        out['pred_noun_3'] = torch.cat(batch_noun_3, dim=0)
        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)

        return out


def create_model(
        model_name: str = None,
        vidx_ridx=None,
        device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'coformer':
        model_config = _C.MODEL.CoFormer
    else:
        raise AttributeError(f'Invalid model_name {model_name}.')
    backbone = build_backbone(
        hidden_dim=model_config.hidden_dim,
        position_embedding=model_config.position_embedding,
        backbone=model_config.backbone,
    )
    transformer = build_transformer(
        d_model=model_config.hidden_dim,
        dropout=model_config.dropout,
        nhead=model_config.nhead,
        num_glance_enc_layers=model_config.num_glance_enc_layers,
        num_gaze_s1_dec_layers=model_config.num_gaze_s1_dec_layers,
        num_gaze_s1_enc_layers=model_config.num_gaze_s1_enc_layers,
        num_gaze_s2_dec_layers=model_config.num_gaze_s2_dec_layers,
        dim_feedforward=model_config.dim_feedforward,
    )
    model = CoFormer(
        backbone,
        transformer,
        num_noun_classes=model_config.num_noun_classes,
        vidx_ridx=vidx_ridx,
    )
    model.to(device)
    return model
