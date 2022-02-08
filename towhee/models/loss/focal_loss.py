# Copyright clcarwin. All rights reserved.
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
# This code is modified by Zilliz.

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Focal loss for object detection.
    Args:
        gamma (`float`): hyper-parameter of focal loss
        alpha (`float`): hyper-parameter of focal loss
        size_average (`bool`): whether to average the loss
    """
    def __init__(self, gamma: float = 0, alpha: float = None, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input_s: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input_s (`float`): predicted result
            target (`float`): ground truth
        """
        if input_s.dim() > 2:
            input_s = input_s.view(input_s.size(0), input_s.size(1), -1)  # N,C,H,W => N,C,H*W
            input_s = input_s.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_s = input_s.contiguous().view(-1, input_s.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input_s)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input_s.data.type():
                self.alpha = self.alpha.type_as(input_s.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
