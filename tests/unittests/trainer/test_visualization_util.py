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
from PIL import Image
from towhee.trainer.utils.visualization import image_folder_sample_show, image_folder_statistic

cur_dir = Path(__file__).parent


class TestVisualizationUtil(unittest.TestCase):
    """
    Test vis
    """

    def setUp(self) -> None:
        self.mock_img = Image.new(mode="RGB", size=(20, 20))
        self.class1_path = Path(cur_dir) / "root" / "class_1"
        self.root_dir = str(Path(cur_dir) / "root")
        Path.mkdir(self.class1_path, parents=True)
        self.img_path = self.class1_path / "mock_img.jpg"
        self.mock_img.save(self.img_path)

    def test_vis(self):
        image_folder_sample_show(root=self.root_dir, rows=1, cols=1, img_size=20)
        self.assertEqual(len(list(self.class1_path.iterdir())), 1)

    def test_statistic(self):
        train_cls_count_dict = image_folder_statistic(self.root_dir, show_bar=True)
        self.assertEqual(len(train_cls_count_dict), 1)

    def tearDown(self) -> None:
        self.img_path.unlink()
        self.class1_path.rmdir()
        Path(self.root_dir).rmdir()


if __name__ == "__main__":
    unittest.main()
