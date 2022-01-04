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

from typing import NamedTuple
import torch

from towhee.operator import Operator


class PytorchTransformerOperator(Operator):
    """
    Pytorch transformer operator
        Args:
            model(nn.Module):
                Model object
            args(Optional[Dict]):
                A dict containing all the other arguments, such as labels_map
    """

    def __init__(self, model, args=None) -> None:
        super().__init__()
        self._model = model
        self._args = args

    def __call__(self, img_tensor) -> NamedTuple('Outputs', [('predict', str)]):
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(img_tensor).squeeze(0)
        k = 1
        if self._args['topk']:
            k = self._args['topk']
        Outputs = NamedTuple('Outputs', [('predict', str)])
        label_l = []
        prob_l = []
        for idx in torch.topk(outputs, k).indices.tolist():
            prob_i = torch.softmax(outputs, -1)[idx].item()
            prob_l.append(prob_i)
            label_i = self._args['labels_map'][idx]
            label_l.append(label_i)
            print(f'[{idx}] {label_i:<75} ({prob_i * 100:.2f}%)')
        return Outputs(label_l[0])
