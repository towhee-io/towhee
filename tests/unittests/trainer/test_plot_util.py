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
import matplotlib

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from towhee.trainer.utils.plot_utils import image_folder_sample_show, image_folder_statistic, show_transform, \
    plot_lrs_for_config, plot_lrs_for_scheduler, interpret_image_classification
from towhee.trainer.training_config import TrainingConfig
from torch import nn
from torch.optim.lr_scheduler import StepLR

cur_dir = Path(__file__).parent
matplotlib.use("agg")


class TestVisualizationUtil(unittest.TestCase):
    """
    Test vis
    """

    def setUp(self) -> None:
        self.mock_img = Image.new(mode="RGB", size=(20, 20))
        self.class1_path = Path(cur_dir) / "root" / "class_1"
        self.root_dir = str(Path(cur_dir) / "root")
        Path.mkdir(self.class1_path, parents=True, exist_ok=True)
        self.img_path = self.class1_path / "mock_img.jpg"
        self.mock_img.save(self.img_path)

    def test_vis(self):
        image_folder_sample_show(root=self.root_dir, rows=1, cols=1, img_size=20)
        self.assertEqual(len(list(self.class1_path.iterdir())), 1)

    def test_statistic(self):
        image_folder_statistic(self.root_dir, show_bar=True)
        self.assertEqual(len(list(self.class1_path.iterdir())), 1)

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

    def test_step_lr(self):
        lr_scheduler = StepLR(self.optimizer, step_size=2, gamma=0.1)
        plot_lrs_for_scheduler(self.optimizer, lr_scheduler, total_steps=10)

    def test_plot_lrs_for_config_when_str(self):
        configs = TrainingConfig()
        plot_lrs_for_config(configs, num_training_steps=20, start_lr=100)

    def test_plot_lrs_for_config_when_dict(self):
        configs = TrainingConfig()
        configs.lr_scheduler_type = {
            "name_": "StepLR",
            "step_size": 2,
            "gamma": 0.1
        }
        plot_lrs_for_config(configs, num_training_steps=20, start_lr=100)

    def test_interpret_image_classification(self):
        self.mock_img = Image.new(mode="RGB", size=(20, 20))
        self.img_path = Path(cur_dir).parent / "mock_img.jpg"
        self.mock_img.save(self.img_path)
        val_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img = Image.open(self.img_path)
        model = resnet18(pretrained=True)
        interpret_image_classification(model, img, val_transform, "IntegratedGradients")
        interpret_image_classification(model, img, val_transform, "Occlusion")
        interpret_image_classification(model, img, val_transform, "GradientShap")
        interpret_image_classification(model, img, val_transform, "Saliency")
        self.img_path.unlink()


if __name__ == "__main__":
    unittest.main()
