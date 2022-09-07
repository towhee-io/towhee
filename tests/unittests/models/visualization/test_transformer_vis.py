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

from pathlib import Path

from towhee.utils.thirdparty.pil_utils import PILImage as Image
from towhee.utils.thirdparty.matplotlib_utils import matplotlib
from towhee.models.visualization.transformer_visualization import show_image_heatmap
from towhee.models import vit
from towhee.models.multiscale_vision_transformers import create_mvit_model

cur_dir = Path(__file__).parent

matplotlib.use("agg")


class TestViTShow(unittest.TestCase):
    """
    test show heatmap for ViT.
    """

    def setUp(self) -> None:
        self.model = vit.create_model(model_name="vit_base_16x224", pretrained=False)

    def test_show_heatmap(self):
        mock_img = Image.new(mode="RGB", size=(20, 20))
        show_image_heatmap(self.model, mock_img)


class TestMViTShow(unittest.TestCase):
    """
    test show heatmap for MViT.
    """

    def setUp(self) -> None:
        self.model = create_mvit_model("imagenet_b_16_conv")

    def test_show_heatmap(self):
        mock_img = Image.new(mode="RGB", size=(20, 20))
        show_image_heatmap(self.model, mock_img)


if __name__ == "__main__":
    unittest.main()
