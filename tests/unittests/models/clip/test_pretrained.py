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

import unittest
import os
from PIL import Image
import torch
from torchvision import transforms
from tests.unittests import VIT_DIR
from towhee.models import clip

try:
    BICUBIC = transforms.InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class TestPretrained(unittest.TestCase):
    """
    Test pretrained CLIP model (jit)
    """
    test_dir = VIT_DIR
    img = Image.open(os.path.join(test_dir, "img.jpg"))
    tfms = transforms.Compose([transforms.Resize(224, interpolation=BICUBIC),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                               ])
    img = tfms(img).unsqueeze(0)
    text = clip.tokenize(["a panda", "a dog"])

    def test_models(self):
        model = clip.create_model(model_name="clip_vit_b32", pretrained=True, jit=True)
        with torch.no_grad():
            logits_per_image, _ = model(self.img, self.text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            self.assertTrue(round(probs[0][0]) == 1)
            self.assertTrue(round(probs[0][1]) == 0)


if __name__ == "__main__":
    unittest.main()
