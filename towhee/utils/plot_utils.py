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

from typing import Union
from pathlib import PosixPath

from towhee.types import Image
from towhee.utils.log import engine_log
from towhee.utils.ndarray_utils import from_src

try:
    from matplotlib import pyplot
except ModuleNotFoundError as e:
    engine_log.error('matplotlib not found, you can install via `pip install matplotlib`.')
    raise ModuleNotFoundError('matplotlib not found, you can install via `pip install matplotlib`.') from e


def plot_img(img: Union[str, PosixPath, Image, list], title: str = None):
    """
    Plot one image in notebook.

    Args:
        img (`Union[str, PosixPath, Image]`):
            Image to plot.
        title (`str`):
            Title of the image to show.
    """
    if isinstance(img, list):
        plot_img_list(img, title)
    else:
        if isinstance(img, Union[str, PosixPath].__args__):
            img = from_src(img)
        _, ax = pyplot.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img.to_ndarray())
        ax.axis('off')
        if title:
            ax.set_title(f'{title:.04f}\n')


def plot_img_list(data: list, titles: list = None):
    """
    Plot image list in notebook.

    Args:
        data (`list`):
            All the dataset.
        titles (`list`):
            Title list of the image to show.
    """
    if len(data) == 1:
        return plot_img(data[0], titles)
    _, axs = pyplot.subplots(1, len(data), figsize=(5 * len(data), 5 * len(data)))
    for ax, i in zip(axs, range(len(data))):
        img = data[i]
        if isinstance(img, Union[str, PosixPath].__args__):
            img = from_src(img)
        ax.imshow(img.to_ndarray())
        ax.axis('off')
        if titles and titles[i]:
            ax.set_title(f'{titles[i]:.04f}\n')
