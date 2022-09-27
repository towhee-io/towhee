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
from torch.nn import functional as F
from loss import ContrastiveLoss

from towhee.models.mcprop.imageextractor import ImageExtractor
from towhee.models.mcprop.textextractor import TextExtractor
from towhee.models.mcprop.featurefusion import FeatureFusion
from towhee.models.mcprop.transformerpooling import TransformerPooling
from towhee.models.mcprop.depthaggregator import DepthAggregator


class Matching(nn.Module):
    """
    Text extractor
    Args:
        common_space_dim (int): common space dimension
        num_text_transformer_layers (int): number of text transformer layers
        img_feat_dim (int): image feature dimension
        txt_feat_dim (int): text feature dimension
        image_disabled (bool): image disabled
        aggregate_tokens_depth (int): aggregate tokens depth
        fusion_mode (str): feature fusion mode
    """
    def __init__(self, common_space_dim, num_text_transformer_layers, img_feat_dim, txt_feat_dim, image_disabled,
                 aggregate_tokens_depth, fusion_mode, text_model, finetune_text_model,
                 image_model, finetune_image_model, max_violation=True, margin=0.2):
        super().__init__()
        self.aggregate_tokens_depth = aggregate_tokens_depth
        self.fusion_mode = fusion_mode
        self.image_disabled = image_disabled
        self.txt_model = TextExtractor(text_model, finetune_text_model)
        if not image_disabled:
            self.img_model = ImageExtractor(image_model, finetune_image_model)
            self.image_fc = nn.Sequential(
                nn.Linear(img_feat_dim, img_feat_dim),
                nn.Dropout(p=0.2),
                # nn.BatchNorm1d(img_feat_dim),
                nn.ReLU(),
                nn.Linear(img_feat_dim, img_feat_dim)
                # nn.BatchNorm1d(img_feat_dim)
            )
            self.process_after_concat = nn.Sequential(      #TODO: for model loading backward compatibility. Move this into FeatureFusionModel
                nn.Linear(img_feat_dim + txt_feat_dim, common_space_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(common_space_dim, common_space_dim)
            ) if self.fusion_mode == 'concat' else FeatureFusion(self.fusion_mode, img_feat_dim, txt_feat_dim, common_space_dim)

        self.caption_process = TransformerPooling(input_dim=txt_feat_dim, output_dim=common_space_dim, num_layers=num_text_transformer_layers)
        self.url_process = TransformerPooling(input_dim=txt_feat_dim, output_dim=txt_feat_dim if not image_disabled else common_space_dim, num_layers=num_text_transformer_layers)
        if self.aggregate_tokens_depth is not None:
            self.token_aggregator = DepthAggregator(self.aggregate_tokens_depth, input_dim=txt_feat_dim, output_dim=common_space_dim)

        self.matching_loss = ContrastiveLoss(margin=margin, max_violation=max_violation)

    def compute_embeddings(self, img, url, url_mask, caption, caption_mask):
        alphas = None
        if torch.cuda.is_available():
            img = img.cuda() if img is not None else None
            url = url.cuda()
            url_mask = url_mask.cuda()
            caption = caption.cuda()
            caption_mask = caption_mask.cuda()

        url_feats = self.txt_model(url, url_mask)
        url_feats_plus = self.url_process(url_feats[-1], url_mask)   # process features from the last layer
        if self.aggregate_tokens_depth:
            url_feats_depth_aggregated = self.token_aggregator(url_feats, url_mask)
            url_feats = url_feats_plus + url_feats_depth_aggregated     # merge together features processed from the last layer and features from the hidden layers of Roberta
        else:
            url_feats = url_feats_plus

        caption_feats = self.txt_model(caption, caption_mask)
        caption_feats_plus = self.caption_process(caption_feats[-1], caption_mask)
        if self.aggregate_tokens_depth:
            caption_feats_depth_aggregated = self.token_aggregator(caption_feats, caption_mask)
            caption_feats = caption_feats_plus + caption_feats_depth_aggregated # same as for urls
        else:
            caption_feats = caption_feats_plus

        if not self.image_disabled:
            # forward img model
            img_feats = self.img_model(img).float()
            img_feats = self.image_fc(img_feats)
            # concatenate img and url features
            if self.fusion_mode == 'concat':
                query_feats = torch.cat([img_feats, url_feats], dim=1)
                query_feats = self.process_after_concat(query_feats)
            else:
                query_feats, alphas = self.process_after_concat(img_feats, url_feats)   # TODO: very bad design for maintaining retro-compatibility
        else:
            query_feats = url_feats

        # L2 normalize output features
        query_feats = F.normalize(query_feats, p=2, dim=1)
        caption_feats = F.normalize(caption_feats, p=2, dim=1)

        return query_feats, caption_feats, alphas

    def compute_loss(self, query_feats, caption_feats):
        loss = self.matching_loss(query_feats, caption_feats)
        return loss

    def forward(self, img, url, url_mask, caption, caption_mask):
        # forward the embeddings
        query_feats, caption_feats, alphas = self.compute_embeddings(img, url, url_mask, caption, caption_mask)

        # compute loss
        loss = self.compute_loss(query_feats, caption_feats)
        return loss, alphas
