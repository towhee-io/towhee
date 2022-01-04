# Copyright 2021 Zilliz. All rights reserved.
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

import unittest
import random
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch import nn
from towhee.models.loss.focal_loss import FocalLoss


class TestOperator(unittest.TestCase):
    """
    A test for focal loss.
    """

    def setUp(self):
        x = torch.rand(128, 1000, 8, 4) * random.randint(1, 10)
        self.input = Variable(x)
        l = torch.rand(128, 8, 4) * 1000  # 1000 is classes_num
        l = l.long()
        self.target = Variable(l)

    def test_sam(self):
        output0 = FocalLoss(gamma=0)(self.input, self.target)
        output1 = nn.NLLLoss2d()(F.log_softmax(self.input), self.target)
        self.assertGreaterEqual(1, abs(output0.item() - output1.item()))
