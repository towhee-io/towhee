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
from torch import nn

import torch
import numpy as np
import cv2


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
