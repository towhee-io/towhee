from typing import List

import torch
from PIL import Image as PILImage

from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


from towhee.operator.model_handler import HandlerBase


class Handler(HandlerBase):
    def __init__(self, model, device_id, **args):
        self._device_id = device_id
        self.model = model
        if device_id >= 0:
            self.model = self.model.to(device_id)
        self.model.eval()
        config = resolve_data_config({}, model=self.model)
        self.tfms = create_transform(**config)

    def preprocess(self, imgs: List['towhee.Image']):
        img_tensors = []
        for img in imgs:
            img = PILImage.fromarray(img.astype('uint8'), 'RGB')
            img = self.tfms(img)
            img_tensors.append(img)
        return torch.stack(img_tensors, 0)

    def postprocess(self, imgs: 'tensor') -> 'numpy.ndarray':
        imgs = torch.unbind(imgs, 0)
        ret = []
        for img in imgs:
            ret.append(img.flatten().detach().numpy())
        return ret

    def __call__(self, imgs: List['towhee.Image']) -> 'numpy.ndarray':
        batch = self.preprocess(imgs)
        if self._device_id >= 0:
            batch = batch.to(self._device_id)
        with torch.no_grad():
            infer = self.model.forward_features(batch)
            if self._device_id >= 0:
                infer = infer.detach().cpu()
        return self.postprocess(infer)
