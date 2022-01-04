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
from pathlib import Path

import torchvision
from torch import nn
from torchvision import transforms

from towhee.cnn_trainer.cnn_trainer import PyTorchCNNTrainer
from towhee.cnn_trainer.training_args import TrainingArguments
from towhee.data.dataset.image_datasets import PyTorchImageDataset

cache_path = Path(__file__).parent.parent.resolve()
image_path = cache_path.joinpath('data/dataset/kaggle_dataset_small/train')
label_file =cache_path.joinpath('data/dataset/kaggle_dataset_small/train/train_labels.csv')


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
        self.training_args = TrainingArguments(
            output_dir='./ResNet50',
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_gpu_train_batch_size=4,
            prediction_loss_only=True,
        )
        self.trainer = PyTorchCNNTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data
        )

    def test_overfit_on_small_batches(self) -> None:
        training_output = self.trainer.train()
        self.assertGreaterEqual(3.0, training_output.training_loss)


if __name__ == '__main__':
    unittest.main(verbosity=1)
