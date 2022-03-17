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
from PIL import Image
from towhee.trainer.utils.visualization import image_folder_sample_show, image_folder_statistic, show_transform, \
    plot_lrs_for_config, plot_lrs_for_scheduler
from towhee.trainer.training_config import TrainingConfig
from torch import nn
from torch.optim.lr_scheduler import StepLR
from unittest import mock

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

    @mock.patch("towhee.trainer.utils.visualization.plt")
    def test_statistic(self, mock_plt):
        image_folder_statistic(self.root_dir, show_bar=True)
        assert mock_plt.show.called

    def test_transform(self):
        show_transform(str(self.img_path), lambda x: x)
        self.assertEqual(len(list(self.class1_path.iterdir())), 1)

    def tearDown(self) -> None:
        self.img_path.unlink()
        self.class1_path.rmdir()
        Path(self.root_dir).rmdir()


class TestVisualizationLRScheduler(unittest.TestCase):
    """
    test lr scheduler plot.
    """

    def setUp(self) -> None:
        model = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=100)

    @mock.patch("towhee.trainer.utils.visualization.plt")
    def test_step_lr(self, mock_plt):
        lr_scheduler = StepLR(self.optimizer, step_size=2, gamma=0.1)
        plot_lrs_for_scheduler(self.optimizer, lr_scheduler, total_steps=10)
        assert mock_plt.show.called

    @mock.patch("towhee.trainer.utils.visualization.plt")
    def test_plot_lrs_for_config_when_str(self, mock_plt):
        configs = TrainingConfig()
        plot_lrs_for_config(configs, num_training_steps=20, start_lr=100)
        assert mock_plt.show.called

    @mock.patch("towhee.trainer.utils.visualization.plt")
    def test_plot_lrs_for_config_when_dict(self, mock_plt):
        configs = TrainingConfig()
        configs.lr_scheduler_type = {
            "name_": "StepLR",
            "step_size": 2,
            "gamma": 0.1
        }
        plot_lrs_for_config(configs, num_training_steps=20, start_lr=100)
        assert mock_plt.show.called


if __name__ == "__main__":
    unittest.main()
