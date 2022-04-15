from typing import List

import torch
from torch import nn

from towhee.operator.model_handler import HandlerBase


class TorchModelHandler(HandlerBase):
    """
    TorchModelWorker
    """

    def __init__(self, model: nn.Module, device_id: int = -1):
        self._device_id = device_id
        self.model = model
        if device_id >= 0:
            self.model = model.to(device_id)

    def preprocess(self, data: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(data, 0)

    def postprocess(self, data: torch.Tensor) -> List[torch.Tensor]:
        return torch.unbind(data, 0)

    def inference(self, data):
        if self._device_id == -1:
            return self._cpu_inference(data)
        else:
            return self._gpu_inference(data)

    def __call__(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        pre_ret = self.preprocess(data)
        infer_ret = self.inference(pre_ret)
        return self.postprocess(infer_ret)

    def _cpu_inference(self, batch_data):
        with torch.no_grad():
            return self.model(batch_data)

    def _gpu_inference(self, batch_data):
        batch_data = batch_data.to(self._device_id)
        with torch.no_grad():
            output = self.model(batch_data)
            return output.detach().cpu()
