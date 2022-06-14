# Built on top of the original implementation at https://github.com/hila-chefer/Transformer-Explainability
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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
from typing import Any

from torch import nn
from PIL import Image

import torch
from torchvision import transforms
import numpy as np
import cv2
from towhee.trainer.utils.file_utils import is_matplotlib_available
from towhee.utils.log import models_log


class LRP:
    """
    Layer-wise relevance propagation.
    https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_lrp(self, input_tensor: nn.Module, index: int = None, method: str = 'transformer_attribution',
                     start_layer: int = 0, device=None):
        """
        Generate LRP for a specific layer.
        Args:
            input_tensor (`nn.Module`):
                The input tensor to the model.
            index (`int`):
                The index of the layer to be visualized.
            method (`str`):
                The method to be used for LRP.
            start_layer (`int`):
                The index of the layer to start from.
            device (`str`):
                Model device.

        Returns:
            (`torch.Tensor`):
                The LRP tensor.
        """
        if device is None:
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
                                  start_layer=start_layer, **kwargs)


def _add_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def _reshape_attr_and_get_heatmap(image_relevance: torch.Tensor, img_tensor: torch.Tensor, img_size=224):
    side_length = int(np.sqrt(image_relevance.shape[-1]))
    image_relevance = image_relevance.reshape(1, 1, side_length, side_length)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=img_size, mode='bilinear')
    image_relevance = image_relevance.reshape(img_size, img_size).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0]
    img_np = img_tensor.permute(1, 2, 0).data.cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    vis = _add_cam_on_image(img_np, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_attention(model: nn.Module, image_tensor: torch.Tensor, method='transformer_attribution',
                       class_index: int = None, img_size: int = 224, device: str = None) -> np.ndarray:
    """
    Generate attention visualization of the model.
    https://openaccess.thecvf.com/content/CVPR2021/papers/
    Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf
    Args:
        model (nn.Module):
            Model to visualize.
        image_tensor (torch.Tensor):
            Image tensor.
        method (str):
            `full`, `rollout` or `transformer_attribution`. Default is `transformer_attribution`.
        class_index (int):
            Class index.
        img_size (int):
            Image size.
        device (str):
            Model device, cpu or cuda

    Returns:
        (`numpy.ndarray`):
            Attention map.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    attribution_generator = LRP(model)
    image_relevance = attribution_generator.generate_lrp(image_tensor.unsqueeze(0).to(device=device),
                                                         method=method,
                                                         index=class_index).detach()
    vis = _reshape_attr_and_get_heatmap(image_relevance, image_tensor, img_size=img_size)
    return vis


def show_image_heatmap(model: nn.Module, pil_img: Image, method: str = 'transformer_attribution', transform: Any = None,
                       device: str = None):
    """
    Show image heatmap for transformer model.
    Args:
        model (`nn.Module`):
            Transformer model.
        pil_img (`Image`):
            Pil Image.
        method (`str`):
            `full`, `rollout` or `transformer_attribution`.
            Algorithm for attention vis, default is `transformer_attribution`
        transform (`Any`):
            Pytorch transforms or function, can be None by default.
        device (`str`):
            model device.

    """
    if transform is None:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    img_tensor = transform(pil_img)
    if not is_matplotlib_available():
        models_log.warning('Matplotlib is not available.')
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(pil_img)
    axs[0].axis('off')
    vis_img = generate_attention(model, img_tensor, method=method, img_size=img_tensor.shape[-1], device=device)
    axs[1].imshow(vis_img)
    axs[1].axis('off')
    plt.show()
