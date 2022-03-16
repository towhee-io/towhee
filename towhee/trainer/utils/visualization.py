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


from typing import Optional, List, Any
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path
from torch import nn
from towhee.trainer.optimization.optimization import get_scheduler, get_warmup_steps
from towhee.trainer.utils.trainer_utils import _construct_scheduler_from_config
from towhee.trainer.training_config import TrainingConfig

import torch
import numpy as np
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


def show_transform(image_path: str, transform: Any):
    """
    Show the result which `torchvision.tranforms` or any other callable function act on the image.
    Only the Image transforms is supported. Such as `ToTensor` or `Normalize` is invalid.
    Args:
        image_path (`str`):
            The original image path.
        transform (`Any`):
            torchvision.tranforms` or any other callable function.

    """
    plt.rcParams['savefig.bbox'] = 'tight'
    orig_img = Image.open(image_path)
    trans_img_list = [transform(orig_img) for _ in range(6)]
    _plot_transform(orig_img, trans_img_list)


def _plot_transform(orig_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(12, 12))
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


def plot_lrs_for_scheduler(optimizer: torch.optim.Optimizer, scheduler: '_LRScheduler', total_steps: int):
    """
    Plot the lr curve for a specified optimizer and scheduler instance.
    Args:
        optimizer (`Optimizer`):
            The optimizer contains the training parameters of a model.
        scheduler (`_LRScheduler`):
            The lr scheduler for an optimizer.
        total_steps (`int`):
            Total training step number.

    """
    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    plt.plot(range(total_steps), lrs)
    plt.show()


def _create_scheduler_from_config(configs: TrainingConfig, num_training_steps: int,
                                  optimizer: torch.optim.Optimizer = None):
    if isinstance(configs.lr_scheduler_type, str):
        lr_scheduler = get_scheduler(
            configs.lr_scheduler_type,
            optimizer=optimizer if optimizer is None else optimizer,
            num_warmup_steps=get_warmup_steps(num_training_steps, configs.warmup_steps, configs.warmup_ratio),
            num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = _construct_scheduler_from_config(optimizer,
                                                        torch.optim.lr_scheduler,
                                                        configs.lr_scheduler_type)
    return lr_scheduler


def plot_lrs_for_config(configs: TrainingConfig, num_training_steps: int, start_lr: int = 100):
    """
    Plot the lr for a config.
    It's useful to visualize how the lr scheduler control the lr by the yaml file configs.
    Args:
        configs (`TrainingConfig`):
            The training config, which can be load from the yaml file.
        num_training_steps (`int`):
            Total training step number.
        start_lr (`int`):
            The start lr value.

    """
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_scheduler = _create_scheduler_from_config(configs, num_training_steps, optimizer)
    plot_lrs_for_scheduler(optimizer, lr_scheduler, num_training_steps)
