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

from towhee.models import video_swin_transformer
from towhee.models.transrac import TransRAC


class TestTransRAC(unittest.TestCase):
    """
    Test TransRAC module
    """
    def test_transrac(self):
        dummy_video = torch.rand(1, 3, 4, 200, 200)  # (bcthw)
        backbone = video_swin_transformer.create_model("swin_t_k400_1k", pretrained=False)
        model = TransRAC(backbone=backbone, num_frames=4)
        out, out_matrix = model(dummy_video)
        self.assertTrue(out.shape == (1, 4))
        self.assertTrue(out_matrix.shape == (1, 12, 4, 4))


if __name__ == "__main__":
    unittest.main()
