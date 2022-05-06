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

from towhee.models.layers.patch_embed3d import PatchEmbed3D


class PatchEmbed3DTest(unittest.TestCase):
    def test_patch_embed3d(self):
        input_tensor = torch.rand(1, 3, 4, 4, 3)
        pe3 = PatchEmbed3D()
        output = pe3(input_tensor)
        self.assertTrue(output.shape == torch.Size([1, 96, 2, 1, 1]))


if __name__ == '__main__':
    unittest.main()
