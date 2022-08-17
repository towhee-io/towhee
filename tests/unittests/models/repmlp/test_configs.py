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

from towhee.models.repmlp import get_configs


class TestConfigs(unittest.TestCase):
    """
    Test RepMLP configs
    """
    def test_configs(self):
        self.assertTrue(get_configs() == get_configs('repmlp_b224'))

        configs1 = get_configs('repmlp_t224')
        self.assertTrue(configs1 == dict(
            channels=(64, 128, 256, 512), hs=(56, 28, 14, 7), ws=(56, 28, 14, 7),
            num_blocks=(2, 2, 6, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 16, 128)
        ))
        configs2 = get_configs('repmlp_t256')
        self.assertTrue(configs2 == dict(
            channels=(64, 128, 256, 512), hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 6, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 16, 128)
        ))
        configs3 = get_configs('repmlp_b224')
        self.assertTrue(configs3 == dict(
            channels=(96, 192, 384, 768), hs=(56, 28, 14, 7), ws=(56, 28, 14, 7),
            num_blocks=(2, 2, 12, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 32, 128)
        ))
        configs4 = get_configs('repmlp_b256')
        self.assertTrue(configs4 == dict(
            channels=(96, 192, 384, 768), hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 12, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 32, 128)
        ))
        configs5 = get_configs('repmlp_d256')
        self.assertTrue(configs5 == dict(
            channels=(80, 160, 320, 640), hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 18, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 16, 128)
        ))
        configs6 = get_configs('repmlp_l256')
        self.assertTrue(configs6 == dict(
            channels=(96, 192, 384, 768), hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 18, 2), reparam_conv_k=(1, 3), sharesets_nums=(1, 4, 32, 256)
        ))


if __name__ == '__main__':
    unittest.main()
