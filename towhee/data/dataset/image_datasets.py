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

import os
import pandas as pd
import torch

from pandas import Series
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple


class PyTorchImageDataset(Dataset):
    """
        PyTorchImageDataset is a dataset class for training.

        Args:
            image_path (:obj:`str`):
                Path to the images for your dataset.

            label_file (:obj:`str`):
                 Path to your label file. The label file should be a csv file. The columns in this file should be
                 [image_name, category], 'image_name' is the path of your images, 'category' is the label of accordance
                 image. For example: [image_name, dog] for one row. Note that the first row should be[image_name, category]

            data_transform (:obj:`Compose`):
                PyTorch transform of the input images.
    """

    def __init__(self, image_path: str, label_file: str, data_transform: transforms.Compose = None):
        self.image_path = image_path
        self.label_file = label_file
        self.data_transform = data_transform

        df = pd.read_csv(self.label_file)
        image_names = Series.to_numpy(df['image_name'])
        for i in range(len(image_names)):
            if os.path.splitext(image_names[i])[1] == '':
                image_names[i] += '.jpg'
        images = image_names.tolist()
        self.images = [os.path.join(self.image_path, i) for i in images]

        categories = Series.to_numpy(df['category'])
        # Count the categories
        breed_set = set(categories)
        breed_list = list(breed_set)
        dic = {}
        for i in range(len(breed_list)):
            dic[breed_list[i]] = i
        self.labels = [dic[categories[i]] for i in range(len(categories))]
        self.num_classes = len(breed_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[index]
        fn = self.images[index]
        img = Image.open(fn)
        if self.data_transform:
            img = self.data_transform(img)

        return (img, label)

    def __len__(self) -> int:
        return len(self.labels)
