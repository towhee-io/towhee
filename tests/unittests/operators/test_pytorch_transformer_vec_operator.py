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

import unittest
from PIL import Image
from torchvision import transforms
import os

from towhee.trainer.models.vit.vit import ViT
from tests.unittests.mock_operators import PYTORCH_TRANSFORMER_VEC_OPERATOR_PATH, load_local_operator
from tests.unittests import VIT_DIR


class TransformerEmbeddingOperatorTest(unittest.TestCase):
    name = 'B_16_imagenet1k'
    test_dir = VIT_DIR
    #weights_path = test_dir + 'B_16_imagenet1k.pth'
    weights_path = None
    model = ViT(name, weights_path=weights_path, pretrained=True, to_vec=True)
    img = Image.open(os.path.join(test_dir, 'img.jpg'))
    tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
    img = tfms(img).unsqueeze(0)

    def test_transformer_embedding_operator(self):
        trans = load_local_operator('pytorch_transformer_vec_operator', PYTORCH_TRANSFORMER_VEC_OPERATOR_PATH)
        op = trans.PytorchTransformerVecOperator(self.model)
        outputs = op(self.img)
        self.assertEqual(outputs.embedding.shape[0], 768)


if __name__ == '__main__':
    unittest.main()
