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

from towhee.models.vit.transformer import merge_last


class TestTransformer(unittest.TestCase):
    """
    Test transformer.
    """
    x = torch.tensor([[[1., -1.], [1., -1.]], [[1., -1.], [1., -1.]]])

    def test_transformer(self):
        y = merge_last(self.x, 2)
        z = torch.tensor([[1., -1., 1., -1.], [1., -1., 1., -1.]])
        c = (y.numpy() == z.numpy())
        self.assertEqual(c.all(), True)


if __name__ == '__main__':
    unittest.main()
