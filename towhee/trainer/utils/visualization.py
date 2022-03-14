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


from typing import Optional, List
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path

import matplotlib.pylab as plt
import random
import os


def _show_grid(row_grid_list, cls_list):
    _, axs = plt.subplots(nrows=len(row_grid_list), ncols=len(row_grid_list[0]), squeeze=False, figsize=(13, 13))
    for row, (row_gird, _) in enumerate(zip(row_grid_list, cls_list)):
        for col, img in enumerate(row_gird):
            show_img = img.numpy().transpose(1, 2, 0)
            axs[row, col].imshow(show_img)
            axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def image_folder_sample_show(root: str, rows: int = 4, cols: int = 4, img_size: int = 224,
                             classes: Optional[List[str]] = None):
    """
    Show sample images for `torchvision.datasets.ImageFolder`,
    Put another way, the image root path structure just need to be like this:
    ```
    root-|
        #classname1#-|
            image-1.jpg
            image-2.jpg
        #classname2-|
            image-1.jpg
            image-2.jpg
        |
        |
        #classnameN-|
            image-1.jpg
            image-2.jpg
    ```
    it's useful to observe the images for different categories.
    You can randomly sample categories, or specify show classes.
    Args:
        root (`str`):
            Root of `ImageFolder` instance.
        rows (`int`):
            Count of show rows.
        cols (`int`):
            Count of show cols.
        img_size (`int` or `tuple`):
            Show image shape, if int, it's a square.
        classes (`Optional[list[str]]`):
            The specified classes you want to see. If not None, param `rows` will be invalid.

    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ])
    root_path = Path(root)
    rows = min(len(list(root_path.iterdir())), rows)
    if classes:
        sample_class_list = [root_path / cls for cls in classes]
    else:
        sample_class_list = random.sample(list(root_path.iterdir()), rows)
    row_grid_list = []
    cls_list = []
    for class_path in sample_class_list:
        cls = class_path.name
        class_sample_num = min(len(list(class_path.iterdir())), cols)
        class_sample_img_list = random.sample(
            list(class_path.iterdir()),
            class_sample_num
        )
        trans_img_list = []
        for img_path in class_sample_img_list:
            img = read_image(str(img_path))
            trans_img = transform(img)
            trans_img_list.append(trans_img)
        row_grid_list.append(trans_img_list)
        cls_list.append(cls)
    _show_grid(row_grid_list, cls_list)


def image_folder_statistic(root: str, classes: Optional[List[str]] = None, show_bar: Optional[bool] = False):
    """
    Show count statistic infos for `torchvision.datasets.ImageFolder`.
    Put another way, the image root path structure just need to be like this:
    ```
    root-|
        #classname1#-|
            image-1.jpg
            image-2.jpg
        #classname2-|
            image-1.jpg
            image-2.jpg
        |
        |
        #classnameN-|
            image-1.jpg
            image-2.jpg
    ```
    Args:
        root (`str`):
            Root of `ImageFolder` instance.
        classes (`Optional[List[str]]`):
            The specified classes of statistic. If not None, all class statistic will be return.
        show_bar (`Optional[bool]`):
            Whether showing bar graph.

    Returns:
        (`dict`)
            Count statistic info, where key is category name, value is the count.
    """
    root_path = Path(root)
    if classes:
        class_list = [root_path / cls for cls in classes]
    else:
        class_list = os.listdir(root)
    count_dict = {}
    for cls in class_list:
        cls_num = len(list((root_path / cls).iterdir()))
        count_dict[cls] = cls_num
    if show_bar:
        plt.figure(figsize=(30, 10))
        plt.bar(x=count_dict.keys(), height=count_dict.values(), color='red')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.grid(True)
        plt.title(root_path.name)
        plt.show()
    return count_dict
