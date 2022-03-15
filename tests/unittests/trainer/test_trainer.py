# coding=utf-8
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
import torchvision
from pathlib import Path
from shutil import rmtree

from torch import nn
from torch.optim import AdamW
from torchvision import transforms
from towhee.operator import NNOperator
from towhee.trainer.training_config import TrainingConfig
from towhee.data.dataset.image_datasets import PyTorchImageDataset

cache_path = Path(__file__).parent.parent.resolve()
image_path = cache_path.joinpath('data/dataset/kaggle_dataset_small/train')
label_file = cache_path.joinpath('data/dataset/kaggle_dataset_small/train/train_labels.csv')


class MockOperator(NNOperator):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

    def __call__(self, *args, **kwargs):
        return 1

    def get_model(self):
        return self.model


class TrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        data_transforms = {
            'train':
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]
                )
        }
        self.train_data = PyTorchImageDataset(str(image_path), str(label_file), data_transforms['train'])
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.train_data.num_classes)
        self.training_cfg = TrainingConfig(
            output_dir='./ResNetOutput', overwrite_output_dir=True, epoch_num=1, batch_size=4, dataloader_num_workers=0,
            tensorboard='null'
        )
        self.op = MockOperator()

    def test_overfit_on_small_batches(self) -> None:
        self.op.train(training_config=self.training_cfg, train_dataset=self.train_data)
        self.assertEqual(1, 1)
        test_path = Path(__file__).parent.joinpath('test_modelcard')
        if test_path.is_dir():
            rmtree(test_path)
        self.op.trainer.save(test_path)
        self.assertTrue(test_path.is_dir())
        self.assertEqual(True, (test_path / 'README.md').exists())
        rmtree(test_path)

    def test_set_loss(self) -> None:
        my_loss = nn.BCELoss()
        loss_name = 'my_loss'
        self.op.trainer.set_loss(my_loss, loss_name=loss_name)
        self.assertEqual(loss_name, self.op.trainer.configs.loss)

    def test_set_optimizer(self) -> None:
        my_optimizer = AdamW(self.op.get_model().parameters(), lr=0.002)
        optimizer_name = 'my_optimizer'
        self.op.trainer.set_optimizer(my_optimizer, optimizer_name=optimizer_name)
        self.assertEqual(optimizer_name, self.op.trainer.configs.optimizer)

    def test_resume_train(self) -> None:
        self.training_cfg.epoch_num = 2
        self.op.train(
            training_config=self.training_cfg,
            train_dataset=self.train_data,
            resume_checkpoint_path=self.training_cfg.output_dir + '/epoch_1'
        )


if __name__ == '__main__':
    unittest.main(verbosity=1)
