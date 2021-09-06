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

from torchvision import transforms
from towhee.dataset.image_datasets import PyTorchImageDataset


class DatasetTest(unittest.TestCase):
    def test_length(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        dataset = PyTorchImageDataset('../kaggle_dataset/train', '../kaggle_dataset/labels.csv', data_transforms['train'])
        images, labels = dataset.images, dataset.labels
        self.assertEqual(len(images), len(labels))


if __name__ == '__main__':
    unittest.main(verbosity=1)
