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
from towhee.models.nnfp import NNFp


class TestFp(unittest.TestCase):
    """
    Test FingerPrint Network
    """
    def test_nnfp(self):
        x = torch.rand(2, 10, 80)
        params = {
            'dim': 8,
            'h': 16,
            'u': 2,
            'fuller': True,
            'activation': 'relu',
            'sample_rate': 80,
            'n_mels': 10,
            'segment_size': 1,
            'hop_length': 1,
        }
        segn = int(params['segment_size'] * params['sample_rate'])
        t = (segn + params['hop_length'] - 1) // params['hop_length']

        model = NNFp(dim=params['dim'], h=params['h'], u=params['u'], in_f=params['n_mels'], in_t=t)
        outs = model(x)
        self.assertEqual(outs.shape, (x.shape[0], params['dim']))


if __name__ == '__main__':
    unittest.main()
