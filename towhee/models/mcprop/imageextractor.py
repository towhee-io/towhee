# Built on top of the original implementation at https://github.com/mesnico/Wiki-Image-Caption-Matching/blob/master/mcprop/model.py
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

import torch
from torch import nn
from towhee.models import clip


class ImageExtractor(nn.Module):
    """
    Image extractor
    Args:
        image_model (str): image model
        finetune (bool): finetune
    """
    def __init__(self, image_model: str, finetune: bool):
        super().__init__()
        self.finetune = finetune
        self.image_model = clip.create_model(image_model, device='cuda' if torch.cuda.is_available() else 'cpu',
                                             pretrained=True)

    def forward(self, img):
        with torch.set_grad_enabled(self.finetune):
            feats = self.image_model.encode_image(img)
        return feats
