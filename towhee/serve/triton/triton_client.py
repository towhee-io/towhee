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
import torch
from typing import List

from towhee.serve.triton.bls.python_backend_wrapper import pb_utils


class TritonClient:
    """
    The TritonClient to access model on Triton server.

    Args:
        model_name (`str`):
            The name of the model.
        input_names (`List[str]`):
            The list of the input names.
        output_names (`List[str]`):
            The list of the output names.
    """
    def __init__(self, model_name: str, input_names: List[str], output_names: List[str]):
        self._model_name = model_name
        self._input_names = input_names
        self._output_names = output_names

    def __call__(self, *args):
        inputs = [
            pb_utils.Tensor(name, array.numpy() if array.device.type == 'cpu' else array.cpu().numpy())
            for name, array in zip(self._input_names, args)
        ]

        inference_request = pb_utils.InferenceRequest(model_name=self._model_name, requested_output_names=self._output_names, inputs=inputs)

        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            if len(self._output_names) == 1:
                outputs = torch.tensor(pb_utils.get_output_tensor_by_name(inference_response, self._output_names[0]).as_numpy())
            else:
                outputs = [torch.tensor(pb_utils.get_output_tensor_by_name(inference_response, name).as_numpy()) for name in self._output_names]

        return outputs
