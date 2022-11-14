# Copyright 2022 Zilliz. All rights reserved.
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

import torch
from torch import nn

from towhee.models.utils.download import download_from_url


def create_model(
        module: nn.Module,
        configs,
        pretrained: bool = False,
        checkpoint_path: str = None,
        device: str = None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    url = None
    if 'url' in configs:
        url = configs['url']
        del configs['url']

    model = module(**configs).to(device)
    if pretrained:
        if checkpoint_path:
            local_path = checkpoint_path
        elif url:
            local_path = download_from_url(url=url)
        else:
            raise AttributeError('No url or checkpoint_path is provided for pretrained model.')
        state_dict = torch.load(local_path, map_location=device)
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
        else:
            model = state_dict
    model.eval()
    return model
