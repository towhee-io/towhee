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
from typing import Optional, List, Any, Tuple, Callable
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path
from torch import nn
from towhee.models import clip
from towhee.models.clip import SimpleTokenizer
from towhee.trainer.optimization.optimization import get_scheduler, get_warmup_steps
from towhee.trainer.utils.trainer_utils import _construct_scheduler_from_config
from towhee.trainer.training_config import TrainingConfig
from towhee.trainer.utils.file_utils import is_captum_available, is_matplotlib_available
from towhee.utils.log import trainer_log

import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import copy


def _show_grid(row_grid_list, cls_list):
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
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
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    plt.rcParams['savefig.bbox'] = 'tight'
    orig_img = Image.open(image_path)
    trans_img_list = [transform(orig_img) for _ in range(sample_num)]
    _plot_transform(orig_img, trans_img_list)


def _plot_transform(orig_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
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


class LRP:
    """
    Layer-wise relevance propagation.
    https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_lrp(self, input_tensor: nn.Module, index: int = None, method: str = 'transformer_attribution',
                     is_ablation: bool = False, start_layer: int = 0):
        """
        Generate LRP for a specific layer.
        Args:
            input_tensor (`nn.Module`):
                The input tensor to the model.
            index (`int`):
                The index of the layer to be visualized.
            method (`str`):
                The method to be used for LRP.
            is_ablation (`bool`):
                Whether to use ablation.
            start_layer (`int`):
                The index of the layer to start from.

        Returns:
            (`torch.Tensor`):
                The LRP tensor.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output = self.model(input_tensor)
        kwargs = {'alpha': 1}
        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
        one_hot = torch.sum(one_hot * output).to(device)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input_tensor.device), method=method,
                                  is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)


def _show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_attention(model: nn.Module, image_tensor: torch.Tensor, class_index: int = None) -> np.ndarray:
    """
    Generate attention visualization of the model.
    https://openaccess.thecvf.com/content/CVPR2021/papers/
    Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf
    Args:
        model (nn.Module):
            Model to visualize.
        image_tensor (torch.Tensor):
            Image tensor.
        class_index (int):
            Class index.

    Returns:
        (`numpy.ndarray`):
            Attention map.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    attribution_generator = LRP(model)
    transformer_attribution = attribution_generator.generate_lrp(image_tensor.unsqueeze(0).to(device=device),
                                                                 method='transformer_attribution',
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = image_tensor.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = _show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_clip_relevance(model: nn.Module, pil_img: Image, text_list: List[str], device: str, vis_start_layer: int = 11,
                       text_start_layer: int = 11, transform: Callable = None, tokenize: Callable = None) -> Tuple:
    """
    Get text relevance and image relevance from CLIP model.
    Args:
        model (`nn.Module`):
            CLIP model to visualize.
        pil_img (`Image`):
            Input image.
        text_list (`List[str]`):
            List of text str.
        device (`str`):
            Device to use.
        vis_start_layer (`int`):
            Start layer for visualization.
        text_start_layer (`int`):
            Start layer for text.
        transform (`Callable`):
            Transform function for image.
        tokenize (`Callable`):
            Tokenize function for text.

    Returns:
        (`Tuple`):
            text_relevance, image_relevance, text_tokens, img_tensor
    """
    if transform is None:
        def _transform(n_px):
            return Compose([
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert('RGB'),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        transform = _transform(224)
    if tokenize is None:
        tokenize = clip.tokenize

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    text_tokens = tokenize(text_list).to(device)
    batch_size = text_tokens.shape[0]
    img_tensors = img_tensor.repeat(batch_size, 1, 1, 1)
    logits_per_image, _ = model(img_tensors, text_tokens)
    index = list(range(batch_size))
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    relevance = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    relevance = relevance.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < vis_start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        relevance = relevance + torch.bmm(cam, relevance)
    image_relevance = relevance[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    rel_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    rel_text = rel_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < text_start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        rel_text = rel_text + torch.bmm(cam, rel_text)
    text_relevance = rel_text
    return text_relevance, image_relevance, text_tokens, img_tensor


def show_image_relevance(image_relevance: torch.Tensor, img_tensor: torch.Tensor, orig_image: Image):
    """
    Show the image relevance heatmap.
    Args:
        image_relevance (`torch.Tensor`):
            Image relevance.
        img_tensor (`torch.Tensor`):
            Transformed image tensor.
        orig_image (`Image`):
            Original input image.
    """
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    side_length = int(np.sqrt(image_relevance.shape[-1]))
    image_relevance = image_relevance.reshape(1, 1, side_length, side_length)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    img_np = img_tensor[0].permute(1, 2, 0).data.cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    vis = _show_cam_on_image(img_np, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis)
    axs[1].axis('off')
    plt.show()


def show_heatmap_on_text(text: str, text_encoding: torch.Tensor, rel_text: torch.Tensor):
    """
    Show the text relevance heatmap.
    Args:
        text (`str`):
            Text to show.
        text_encoding (`torch.Tensor`):
            Tokenized text.
        rel_text (`torch.Tensor`):
            Text relevance.

    """
    if not is_captum_available():
        trainer_log.warning('You should install Captum first. Please run `pip install captum`.')
        return None
    from captum.attr import visualization as viz  # pylint: disable=import-outside-toplevel
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel

    tokenizer = SimpleTokenizer()

    cls_idx = text_encoding.argmax(dim=-1)
    rel_text = rel_text[cls_idx, 1:cls_idx]
    text_scores = rel_text / rel_text.sum()
    text_scores = text_scores.flatten()
    text_tokens = tokenizer.encode(text)
    text_tokens_decoded = [tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [viz.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
    viz.visualize_text(vis_data_records)
    plt.show()


def show_attention_for_clip(model: nn.Module, pil_img: Image, text_list: List[str], device: str = None):
    """
    Show the attention for CLIP model. This function can show one image with multiple texts.
    Args:
        model (`nn.Module`):
            CLIP model to show attention.
        pil_img (`Image`):
            Image to show.
        text_list (`List[str]`):
            Text list to show.
        device (`str`):
            Device to use.

    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rel_text, rel_image, text_tokens, img_tensor = get_clip_relevance(model, pil_img, text_list, device)
    if not is_matplotlib_available():
        trainer_log.warning('Matplotlib is not available.')
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    batch_size = len(text_list)
    for i in range(batch_size):
        show_heatmap_on_text(text_list[i], text_tokens[i], rel_text[i])
        show_image_relevance(rel_image[i], img_tensor, orig_image=pil_img)
        plt.show()
