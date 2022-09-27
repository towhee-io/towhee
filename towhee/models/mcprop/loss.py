# Built on top of the original implementation at https://github.com/mesnico/Wiki-Image-Caption-Matching/blob/master/mcprop/loss.py
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
from torch import nn as nn


def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class Contrastive(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super().__init__()
        self.margin = margin
        if measure == 'order':
            # self.sim = order_sim
            raise NotImplementedError
        elif measure == 'cosine':
            # self.sim = cosine_sim
            raise NotImplementedError
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def compute_contrastive_loss(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        i = mask
        if torch.cuda.is_available():
            i = i.cuda()
        cost_s = cost_s.masked_fill_(i, 0)
        cost_im = cost_im.masked_fill_(i, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    Args:
        margin (int): margin
        max_violation (int): image feature dimension
    """

    def __init__(self, margin=0, max_violation=False):
        super().__init__()
        self.sim = dot_sim

    def forward(self, im, s, return_similarity_mat=False):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        loss = self.compute_contrastive_loss(scores)
        if return_similarity_mat:
            return loss, scores
        else:
            return loss
