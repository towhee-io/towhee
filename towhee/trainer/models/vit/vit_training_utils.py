# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Pytorch impletation of 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'.
Jax version: https://github.com/google-research/vision_transformer
"""

import numpy as np
import torch
from torch.utils import model_zoo
from scipy.ndimage import zoom


from towhee.trainer.models.vit.vit_pretrained import PRETRAINED_MODELS


def load_pretrained_weights(
    model,
    model_name=None,
    weights_path=None,
    load_first_conv=True,
    load_fc=True,
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """
    Load pretrained weights from path or url.
    Args:
        model(nn.Module):
            Model.
        model_name(str):
            Model name
        weights_path(str, optional):
            Path to pretrained weights file on the local disk.
        load_first_conv(bool):
            If patch embedding is loaded.
        load_fc(bool):
            If pretrained weights of fc layer is loaded at the end of the model.
        load_repr_layer(bool)
            If representation is loaded.
        resize_positional_embedding(bool):
            If positional embedding is resized.
        verbose (bool):
            If printing is done when downloading is over.
    """
    print('model_name is '+str(model_name))
    print('weights_path is '+str(weights_path))
    assert bool(model_name), 'Expected model_name'

    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            print('Please check hub connection in case weights can not be downloaded!')
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} is not available.')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding:
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new,
                has_class_token=hasattr(model, 'class_token'))
        maybe_print(f'Resized positional embeddings from {posemb.shape} to {posemb_new.shape}', verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    print(f'ret is {ret}')
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            f'Missing keys when loading pretrained weights: {ret.missing_keys}'
        assert not ret.unexpected_keys, \
            f'Missing keys when loading pretrained weights: {ret.unexpected_keys}'
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print(f'Missing keys when loading pretrained weights: {ret.missing_keys}', verbose)
        maybe_print(f'Unexpected keys when loading pretrained weights: {ret.unexpected_keys}', verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """
    Rescale the grid of position embeddings in a sensible manner
    """

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb
