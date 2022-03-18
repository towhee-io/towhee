from typing import List

import torch
from torch import nn

def default_batch_callback(data: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(data, 0)

def default_unstack_callback(data:torch.Tensor) -> List[torch.Tensor]:
    return torch.unbind(data, 0)

class TorchModelWorker():
    """
    TorchModelWorker
    """
    def __init__(self, model: nn.Module, device_id: int=-1, batch_callback=default_batch_callback, unstack_callback=default_unstack_callback):
        self._batch_callback = batch_callback
        self._unstack_callback = unstack_callback
        self._device_id = device_id
        self.model = model
        if device_id >= 0:
            self.model = model.to(device_id)

    def __call__(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_data = self._batch_callback(data)
        if self._device_id >= 0:
            output = self._gpu_inference(batch_data)
        else:
            output = self._cpu_inference(batch_data)
        return self._unstack_callback(output)

    def _cpu_inference(self, batch_data):
        return self.model(batch_data)

    def _gpu_inference(self, batch_data):
        batch_data = batch_data.to(self._device_id)
        output = self.model(batch_data)
        return output.detach().cpu()
