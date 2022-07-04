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

import unittest
import torch

from towhee.models.coformer.coformer import create_model
from towhee.models.coformer.utils import nested_tensor_from_tensor_list
from towhee.models.coformer.config import vidx_ridx

class CoFormertTest(unittest.TestCase):
    def test_CoFormer(self):
        """
        Test CoFormer model.
        """
        model = create_model(model_name = 'coformer', vidx_ridx = vidx_ridx)
        x = torch.randn(1,3,40,40)
        x = nested_tensor_from_tensor_list(x)
        output = model(x,inference=True)
        pred_verb = output['pred_verb'][0]
        pred_noun = output['pred_noun_3'][0]
        pred_bbox = output['pred_bbox'][0]
        pred_bbox_conf = output['pred_bbox_conf'][0]
        self.assertTrue(pred_verb.shape == torch.Size([504]))
        self.assertTrue(pred_noun.shape == torch.Size([6, 9929]))
        self.assertTrue(pred_bbox.shape == torch.Size([6, 4]))
        self.assertTrue(pred_bbox_conf.shape == torch.Size([6, 1]))
