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

from towhee.models import acar_net


class TestBackbone(unittest.TestCase):
    """
    Test ACAR-NET Backbone
    """
    data = torch.rand(1, 3, 8, 4, 4)

    def test_slowfast50(self):
        model = acar_net.backbone(
            'slowfast50',
            alpha=4, beta=0.125, fuse_only_conv=False, fuse_kernel_size=7, slow_full_span=True)
        out = model(self.data)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (1, 2048, 2, 1, 1))
        self.assertEqual(out[1].shape, (1, 256, 8, 1, 1))

    def test_slowfast101(self):
        model = acar_net.backbone('slowfast101', alpha=4, beta=0.125)
        out = model(self.data)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (1, 2048, 2, 1, 1))
        self.assertEqual(out[1].shape, (1, 256, 8, 1, 1))

    def test_slowfast152(self):
        model = acar_net.backbone('slowfast152', alpha=4, beta=0.125)
        out = model(self.data)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (1, 2048, 2, 1, 1))
        self.assertEqual(out[1].shape, (1, 256, 8, 1, 1))

    def test_slowfast200(self):
        model = acar_net.backbone('slowfast200', alpha=4, beta=0.125)
        out = model(self.data)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (1, 2048, 2, 1, 1))
        self.assertEqual(out[1].shape, (1, 256, 8, 1, 1))


if __name__ == '__main__':
    unittest.main()
