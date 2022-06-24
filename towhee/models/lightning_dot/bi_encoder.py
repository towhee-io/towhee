# Built on top of the original implementation at https://github.com/intersun/LightningDOT
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
from typing import Any

import torch

from collections import defaultdict
from torch import nn


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, UniterEncoder: nn.Module, BertEncoder: nn.Module, args: Any,  # pylint: disable=invalid-name
                 fix_img_encoder: bool = False, fix_txt_encoder: bool = False, project_dim: int = 0):
        super().__init__()
        if args.img_model_type == 'uniter-base':
            self.img_model = UniterEncoder.init_encoder(args.img_model_config, checkpoint_path=args.img_checkpoint,
                                                        project_dim=project_dim)
        else:
            raise ValueError(f'image encoder does not support other types ({args.img_model_type}) for now')

        if args.txt_model_type == 'bert-base':
            self.txt_model = BertEncoder.init_encoder(args.txt_model_config, checkpoint_path=args.txt_checkpoint,
                                                      project_dim=project_dim)
        elif args.txt_model_type == 'uniter-base':
            self.txt_model = UniterEncoder.init_encoder(args.txt_model_config, checkpoint_path=args.txt_checkpoint,
                                                        project_dim=project_dim)
        else:
            raise ValueError(f'txt encoder does not support other types ({args.txt_model_type}) for now')

        self.fix_img_encoder = fix_img_encoder
        self.fix_txt_encoder = fix_txt_encoder
        self.project_dim = project_dim
        if fix_txt_encoder:
            for param in self.txt_model.parameters():
                param.requires_grad = False
        if fix_img_encoder:
            for param in self.img_model.parameters():
                param.requires_grad = False

    @staticmethod
    def get_representation(sub_model, input_ids, attention_mask, position_ids, img_feat, img_pos_feat, img_masks,
                           gather_index=None, fix_encoder=False):
        if fix_encoder:
            with torch.no_grad():
                sequence_output, pooled_output, hidden_states = sub_model(input_ids, attention_mask, position_ids,
                                                                          img_feat, img_pos_feat, img_masks,
                                                                          gather_index)
        else:
            sequence_output, pooled_output, hidden_states = sub_model(input_ids, attention_mask, position_ids,
                                                                      img_feat, img_pos_feat, img_masks,
                                                                      gather_index)

        if sub_model.training:
            sequence_output.requires_grad_(requires_grad=True)
            pooled_output.requires_grad_(requires_grad=True)

        return sequence_output, pooled_output, hidden_states

    def forward(self, batch, output_all_encoded_layers=False):
        # batch keys
        #   imgs
        #   txts
        #   caps
        batch = defaultdict(lambda: None, batch)

        if 'txts' in batch:
            sb = batch['txts']
            txt_seq, txt_pooled, _ = self.get_representation(self.txt_model, sb['input_ids'],
                                                             sb['attention_mask'], sb['position_ids'],
                                                             sb['img_feat'], sb['img_pos_feat'],
                                                             sb['img_masks'],
                                                             sb['gather_index'], self.fix_txt_encoder)
        else:
            txt_seq, txt_pooled = None, None

        if 'imgs' in batch:
            sb = batch['imgs']
            img_seq, img_pooled, _ = self.get_representation(self.img_model, sb['input_ids'],
                                                             sb['attention_mask'], sb['position_ids'],
                                                             sb['img_feat'], sb['img_pos_feat'],
                                                             sb['img_masks'],
                                                             sb['gather_index'], self.fix_txt_encoder)
        else:
            img_seq, img_pooled = None, None

        if 'caps' in batch and batch['caps']['input_ids'] is not None:
            sb = batch['caps']
            cap_seq, cap_pooled, _ = self.get_representation(self.txt_model, sb['input_ids'],
                                                             sb['attention_mask'], sb['position_ids'],
                                                             sb['img_feat'], sb['img_pos_feat'],
                                                             sb['img_masks'],
                                                             sb['gather_index'], self.fix_txt_encoder)
        else:
            cap_seq, cap_pooled = None, None

        if output_all_encoded_layers:
            return txt_seq, img_seq, cap_seq
        else:
            return txt_pooled, img_pooled, cap_pooled
