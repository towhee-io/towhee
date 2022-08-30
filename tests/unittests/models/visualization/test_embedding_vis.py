# Copyright 2022 Zilliz. All rights reserved.
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

from pathlib import Path

import torch

from torchvision.models import resnet18
from torchvision import transforms

from towhee.utils.pil_utils import PILImage as Image
from towhee.utils.matplotlib_utils import matplotlib
from towhee.models.embedding.embedding_extractor import EmbeddingExtractor
from towhee.models.visualization.embedding_visualization import show_embeddings

cur_dir = Path(__file__).parent


matplotlib.use("agg")


class TestShowEmbeddings(unittest.TestCase):
    """
    test Embedding Visualization
    """

    def test_show_embeddings(self):
        device = torch.device("cpu")
        model = resnet18(pretrained=True)
        model = model.to(device)
        emb_extractor = EmbeddingExtractor(model)
        layer_name_list = [  # "conv1",
            "layer1.0.conv1",
        ]
        for layer_name in layer_name_list:
            emb_extractor.register(layer_name)
        image = Image.new(mode="RGB", size=(20, 20))
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = transform(image).unsqueeze(dim=0).to(device)
        model(img)
        show_embeddings(emb_extractor.emb_out.embeddings, figsize=(10, 80), emb_name_list=layer_name_list)
        for emb in emb_extractor.emb_out.embeddings:
            print(emb.shape)


if __name__ == "__main__":
    unittest.main()
