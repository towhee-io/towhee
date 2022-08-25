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

from typing import List, Tuple

import torch
from towhee.utils.log import models_log
from towhee.trainer.utils.file_utils import is_matplotlib_available
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-import


def show_embeddings(emb_list: List[torch.Tensor], figsize: Tuple = (10, 10), emb_name_list: [List[str]] = None):
    """
    Show layer embeddings
    Args:
        emb_list (`List[torch.Tensor]`):
            Layer embedding List, which can cantains many channels.
        figsize (`Tuple`):
            Figure size.
        emb_name_list (`List[str]`):
            Embedding name list.

    """
    if not is_matplotlib_available():
        models_log.warning('Matplotlib is not available.')
    from towhee.utils.matplotlib_utils import matplotlib
    import matplotlib.pylab as plt  # pylint: disable=import-outside-toplevel
    if isinstance(emb_name_list, list):
        assert len(emb_name_list) == len(
            emb_list), f'len(emb_name_list) is {len(emb_name_list)}, and len(emb_list) is {len(emb_list)}, they must be the same.'
    max_dim = 1
    new_emb_list = []
    for emb in emb_list:
        #  (B=1, C, H, W) or (C, H, W)
        assert (len(emb.shape) == 4 and emb.shape[0] == 1) or len(emb) == 3
        if len(emb.shape) == 4:
            new_emb = torch.squeeze(emb, dim=0).detach()
        else:
            new_emb = emb.detach()
        if new_emb.shape[0] > max_dim:
            max_dim = new_emb.shape[0]
        new_emb_list.append(new_emb)
    _, axs = plt.subplots(nrows=max_dim + 1, ncols=len(new_emb_list), squeeze=False, figsize=figsize)
    for row in range(max_dim + 1):
        dim_idx = row - 1
        for col in range(len(new_emb_list)):
            emb_layer_idx = col
            if row == 0:
                if emb_name_list is not None:
                    axs[0, col].text(0.5, 0.1, f'{emb_name_list[emb_layer_idx]}', horizontalalignment='center',
                                     verticalalignment='center', transform=axs[0, col].transAxes)
                axs[0, col].axis('off')
            else:
                if dim_idx >= new_emb_list[emb_layer_idx].shape[0]:
                    axs[row, col].axis('off')
                    continue
                one_channel_emb = torch.unsqueeze(new_emb_list[emb_layer_idx][dim_idx], dim=-1)
                emb_np = one_channel_emb.numpy()
                emb_np = (emb_np - emb_np.min()) / (emb_np.max() - emb_np.min()) * 255
                axs[row, col].imshow(emb_np)
            axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
