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

from towhee.models.mpvit import create_model


class TestMPViT(unittest.TestCase):
    """
    Test MPViT model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 5
    model = create_model(model_name="mpvit_tiny", pretrained=False, device=device, num_classes=num_classes)
    model.eval()

    def test_forward(self):
        img = torch.rand(1, 3, 224, 224).to(self.device)
        out = self.model(img)
        print(out.shape)
        self.assertTrue(out.shape == (1, self.num_classes))


if __name__ == "__main__":
    unittest.main()
