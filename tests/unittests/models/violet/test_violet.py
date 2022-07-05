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
import numpy as np

from towhee.models.violet.violet import VioletBase


class VioletTest(unittest.TestCase):
    """
    Simple operator test
    """
    def test_violet(self):
        img = torch.rand(1, 5, 3, 32, 32)
        txt = torch.randint(10, size=(1, 5,))
        mask_i = [[1, 1, 1, 0, 0]]
        mask = []
        for i in range(0, 1):
            mask.append(mask_i)
        img = torch.from_numpy(np.array(img, dtype=np.float32))
        txt = torch.from_numpy(np.array(txt, dtype=np.int))
        mask = torch.from_numpy(np.array(mask, dtype=np.int))

        md = VioletBase()
        feat_img, _, _, _ = md.go_feat(img, txt, mask)
        self.assertTrue(feat_img.shape == torch.Size([1, 10, 768]))


if __name__ == "__main__":
    unittest.main()
