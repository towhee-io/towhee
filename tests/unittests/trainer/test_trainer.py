# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team and 2021 Zilliz.
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
import os
from pathlib import Path
from torch import nn
from torch.optim import AdamW
from torchvision import transforms
from towhee.operator import NNOperator
from towhee.trainer.training_config import TrainingConfig
from towhee.data.dataset.image_datasets import PyTorchImageDataset

cache_path = Path(__file__).parent.parent.resolve()
image_path = cache_path.joinpath('data/dataset/kaggle_dataset_small/train')
label_file =cache_path.joinpath('data/dataset/kaggle_dataset_small/train/train_labels.csv')


class MockOperator(NNOperator):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
    def __call__(self, *args, **kwargs):
        return 1
    def get_model(self):
        return self.model

# op = MockOperator()
class TrainerTest(unittest.TestCase):


    def setUp(self) -> None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.train_data = PyTorchImageDataset(image_path, label_file, data_transforms['train'])
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.train_data.num_classes)
        self.training_args = TrainingConfig(
            output_dir='./ResNet50',
            overwrite_output_dir=True,
            epoch_num=1,
            batch_size=4,
            dataloader_num_workers=0,
            tensorboard='null'
        )
        self.op = MockOperator()
        self.op.train(training_config=self.training_args, train_dataset=self.train_data)
        self.training_args.epoch_num = 2
        self.op.train(
            training_config=self.training_args,
            train_dataset=self.train_data,
            resume_checkpoint_path=self.training_args.output_dir + '/epoch_1'
        )
        # self.trainer = Trainer(
        #     operator=op,
        #     training_config=self.training_args,
        #     train_dataset=self.train_data
        # )


    def test_overfit_on_small_batches(self) -> None:
        self.op.train(training_config=self.training_args,train_dataset=self.train_data)
        self.assertEqual(1, 1)
        test_path = Path(__file__).parent.joinpath('test_modelcard')
        self.op.trainer.save(test_path)
        self.assertEqual(True, os.path.exists(test_path.joinpath('README.md')))

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


if __name__ == '__main__':
    unittest.main(verbosity=1)
