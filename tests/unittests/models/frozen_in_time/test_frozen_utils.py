# Copyright 2021 Zilliz. All rights reserved.
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
from towhee.models.frozen_in_time.frozen_utils import sim_matrix, state_dict_data_parallel_fix


class ForzenUtilsTest(unittest.TestCase):

    def test_sim_matrix(self):
        '''
        test the sim_matrix function
        Returns:None
        '''
        x = torch.randn(1, 24)
        y = torch.randn(1, 24)
        out = sim_matrix(x, y)
        self.assertEqual(out.shape, (1, 1))

    def test_state_dict_data_parallel_fix(self):
        '''
        test the state_dict_data_parallel_fix function
        Returns: None
        '''
        x = {'module.text_model.embeddings.word_embeddings.weight': torch.randn(2, 4),
             'module.text_model.embeddings.position_embeddings.weight': torch.randn(2, 4)}
        y = {'text_model.embeddings.word_embeddings.weight': torch.randn(2, 4),
             'text_model.embeddings.position_embeddings.weight': torch.randn(2, 4)}
        out = state_dict_data_parallel_fix(x, y)
        self.assertEqual(y.keys(), out.keys())


if __name__ == '__main__':
    unittest.main()
