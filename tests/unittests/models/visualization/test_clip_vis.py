# Copyright 2022 Zilliz. All rights reserved.
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

from pathlib import Path


from towhee.utils.pil_utils import PILImage as Image
from towhee.utils.matplotlib_utils import matplotlib
from towhee.models.visualization.clip_visualization import show_attention_for_clip
from towhee.models.clip import clip

cur_dir = Path(__file__).parent
matplotlib.use("agg")


class TestCLIPAttention(unittest.TestCase):
    """
    test CLIP attention visualization
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    def test_attention(self):
        model = clip.create_model(
            model_name="clip_vit_b32",
            pretrained=False,  # Use True in practice
            device=self.device,
            jit=False,
            vis=True
        )
        mock_img = Image.new(mode="RGB", size=(20, 20))
        text_list = ["This is a mock text."]
        print(model)
        show_attention_for_clip(model, mock_img, text_list, device=self.device)


if __name__ == "__main__":
    unittest.main()
