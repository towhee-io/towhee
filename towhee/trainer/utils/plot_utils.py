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
from typing import Optional, List, Any, Tuple
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path
from torch import nn
from towhee.utils.thirdparty.pil_utils import PILImage as Image
from towhee.trainer.optimization.optimization import get_scheduler, get_warmup_steps
from towhee.trainer.utils.trainer_utils import _construct_scheduler_from_config
from towhee.trainer.training_config import TrainingConfig
from towhee.trainer.utils.file_utils import is_captum_available, is_matplotlib_available
from towhee.utils.log import trainer_log

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-import


def _show_grid(row_grid_list, cls_list):
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    from towhee.utils.thirdparty.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel

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
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    from towhee.utils.thirdparty.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel

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


def show_transform(image_path: str, transform: Any, sample_num: int = 6):
    """
    Show the result which `torchvision.transforms` or any other callable function act on the image.
    Only the Image transforms is supported. Such as `ToTensor` or `Normalize` is invalid.
    Args:
        image_path (`str`):
            The original image path.
        transform (`Any`):
            torchvision.transforms` or any other callable function.
        sample_num (`int`):
            Sample number.

    """
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    from towhee.utils.thirdparty.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    plt.rcParams['savefig.bbox'] = 'tight'
    orig_img = Image.open(image_path)
    trans_img_list = [transform(orig_img) for _ in range(sample_num)]
    _plot_transform(orig_img, trans_img_list)


def _plot_transform(orig_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    from towhee.utils.thirdparty.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
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
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    from towhee.utils.thirdparty.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    plt.plot(range(total_steps), lrs)
    plt.show()


def _create_scheduler_from_config(configs: TrainingConfig, num_training_steps: int,
                                  optimizer: torch.optim.Optimizer = None):
    assert isinstance(configs.lr_scheduler_type, (str, dict))
    lr_scheduler = None
    if isinstance(configs.lr_scheduler_type, str):
        lr_scheduler = get_scheduler(
            configs.lr_scheduler_type,
            optimizer=optimizer if optimizer is None else optimizer,
            num_warmup_steps=get_warmup_steps(num_training_steps, configs.warmup_steps, configs.warmup_ratio),
            num_training_steps=num_training_steps,
        )
    elif isinstance(configs.lr_scheduler_type, dict):
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


def predict_image_classification(model: nn.Module, input_: torch.Tensor):
    """
    Predict using an image classification model.
    Args:
        model (`nn.Module`):
            Pytorch model.
        input_ (`Tensor`):
            Input image tensor.
    Returns:
        (`tuple`)
            Prediction score which max is 1, and label idx.

    """
    output = model(input_)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    if isinstance(pred_label_idx, torch.Tensor):
        pred_label_idx = pred_label_idx.squeeze().item()
    prediction_score = prediction_score.squeeze().detach().item()
    return prediction_score, pred_label_idx


def interpret_image_classification(model: nn.Module, image: Any, eval_transform: transforms.Compose, method: str,
                                   fig_size: Tuple = (10, 10), cmap: Any = 'OrRd', pred_label_idx: int = None,
                                   titles: List = None, **kwargs: Any):
    """
    Use Captum to interpret the specified class of network output.
    Captum should be installed.
    Args:
        model (`nn.Module`):
            Pytorch module.
        image (`Any`):
            The image before do transform.
            It can be produced by either pytorch `DataLoader` or read by `Image.open()` using PIL.
        eval_transform (`transforms.Compose`):
            Evaluation transform.
        method (`method`):
            It can be in ['Occlusion', 'IntegratedGradients', 'GradientShap', 'Saliency'].
        fig_size (`Tuple`):
            Figure plotting size.
        cmap (`Any`):
            Matplotlib colormap.
        pred_label_idx (`int`):
            If None, use the predicted class automatically.
        titles (`List`):
            Plotted titles of the two axs in the figure.
        **kwargs (`Any`):
            Keyword Args.
    Returns:
        (`tuple`)
            Prediction score and label idx. If input label is specified, the prediction score is None.
    """
    if not is_captum_available():
        trainer_log.warning('You should install Captum first. Please run `pip install captum`.')
        return None
    from captum.attr import visualization as viz  # pylint: disable=import-outside-toplevel
    assert method in ['Occlusion', 'IntegratedGradients', 'GradientShap', 'Saliency']

    def _viz_image_attr_multiple(attribution, trans_image: torch.Tensor, fig_size, cmap,
                                 titles: List[str] = None, **kwargs):
        _ = viz.visualize_image_attr_multiple(
            np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(trans_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ['original_image', 'blended_heat_map'],
            ['all', 'positive'],
            show_colorbar=True,
            outlier_perc=2,
            fig_size=fig_size,
            cmap=cmap,
            titles=titles,
            **kwargs
        )

    model.eval()

    #  prepare image and input
    input_ = eval_transform(image)
    trans_without_norm = copy.deepcopy(eval_transform)
    for trans in trans_without_norm.transforms:
        if trans.__class__.__name__ == 'Normalize':
            trans_without_norm.transforms.remove(trans)
    trans_image = trans_without_norm(image)
    if len(trans_image.shape) == 3:
        trans_image = trans_image.unsqueeze(0)
    if len(input_.shape) == 3:
        input_ = input_.unsqueeze(0)

    prediction_score = None
    if pred_label_idx is None:
        prediction_score, pred_label_idx = predict_image_classification(model, input_)

    if titles is None:
        titles = ['Origin Image', 'Method:' + method]

    if method == 'Occlusion':
        from captum.attr import Occlusion  # pylint: disable=import-outside-toplevel
        occlusion = Occlusion(model)

        attributions_occ = occlusion.attribute(input_,
                                               strides=(3, 8, 8),
                                               target=pred_label_idx,
                                               sliding_window_shapes=(3, 15, 15),
                                               baselines=0)
        _viz_image_attr_multiple(attributions_occ, trans_image, fig_size, cmap, titles=titles, **kwargs)

    if method == 'IntegratedGradients':
        from captum.attr import NoiseTunnel  # pylint: disable=import-outside-toplevel
        from captum.attr import IntegratedGradients  # pylint: disable=import-outside-toplevel
        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(input_, nt_samples=10, nt_type='smoothgrad_sq',
                                                    target=pred_label_idx)
        _viz_image_attr_multiple(attributions_ig_nt, trans_image, fig_size, cmap, titles=titles, **kwargs)

    if method == 'GradientShap':
        from captum.attr import GradientShap, NoiseTunnel  # pylint: disable=import-outside-toplevel
        gradient_shap = GradientShap(model)
        rand_img_dist = torch.cat([input_ * 0, input_ * 1])
        attributions_gs = gradient_shap.attribute(input_,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        _viz_image_attr_multiple(attributions_gs, trans_image, fig_size, cmap, titles=titles, **kwargs)

    if method == 'Saliency':
        from captum.attr import Saliency  # pylint: disable=import-outside-toplevel
        saliency = Saliency(model)
        grads = saliency.attribute(input_, target=pred_label_idx)
        _viz_image_attr_multiple(grads, trans_image, fig_size, cmap, titles=titles, **kwargs)

    return prediction_score, pred_label_idx
