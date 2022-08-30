# Built on top of the original implementation at https://github.com/hila-chefer/Transformer-MM-Explainability
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
from typing import List, Tuple, Callable
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch import nn
from towhee.utils.pil_utils import PILImage as Image
from towhee.models.visualization.transformer_visualization import _reshape_attr_and_get_heatmap
from towhee.models import clip
from towhee.models.clip import SimpleTokenizer
from towhee.trainer.utils.file_utils import is_captum_available, is_matplotlib_available
from towhee.utils.log import models_log

import torch
import numpy as np
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-import


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
    logits_per_image, _ = model(img_tensors, text_tokens, device=device)
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
        models_log.warning('Matplotlib is not available.')
    from towhee.utils.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    vis = _reshape_attr_and_get_heatmap(image_relevance, img_tensor)
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
        models_log.warning('You should install Captum first. Please run `pip install captum`.')
        return None
    from captum.attr import visualization as viz  # pylint: disable=import-outside-toplevel
    if not is_matplotlib_available():
        models_log.warning('Matplotlib is not available.')
    from towhee.utils.matplotlib_utils import matplotlib
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
        models_log.warning('Matplotlib is not available.')
    from towhee.utils.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    batch_size = len(text_list)
    for i in range(batch_size):
        show_heatmap_on_text(text_list[i], text_tokens[i], rel_text[i])
        show_image_relevance(rel_image[i], img_tensor, orig_image=pil_img)
        plt.show()
