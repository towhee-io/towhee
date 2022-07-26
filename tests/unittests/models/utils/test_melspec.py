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
import torch

from towhee.models.utils.melspec import build_mel_spec


class TestMelSpec(unittest.TestCase):
    """
    Test mel spec layer.
    """
    audio = torch.rand(2, 120)

    def test_create_melspec(self):
        params = {
            'sample_rate': 100,
            'window_length': 12, 'hop_length': 10,
            'f_min': 3, 'f_max': 50, 'n_mels': 10,
            'naf_mode': False, 'mel_log': 'log', 'spec_norm': 'l2'
        }
        mel = build_mel_spec(params)

        outs = mel(self.audio)
        self.assertTrue(outs.shape == (2, 10, 13))

    def test_naf_log10_max(self):
        params = {
            'sample_rate': 100,
            'window_length': 12, 'hop_length': 10,
            'f_min': 3, 'f_max': 50, 'n_mels': 10,
            'naf_mode': True, 'mel_log': 'log10', 'spec_norm': 'max'
        }
        mel = build_mel_spec(params)

        outs = mel(self.audio)
        self.assertTrue(outs.shape == (2, 10, 13))


if __name__ == '__main__':
    unittest.main(verbosity=1)
