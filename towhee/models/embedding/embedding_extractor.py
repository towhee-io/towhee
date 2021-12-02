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


class EmbeddingOutput:
    """
    Container for embedding extractor.
    """
    def __init__(self):
        self.embeddings = []

    def __call__(self, module, module_in, module_out):
        self.embeddings.append(module_out)

    def clear(self):
        self.embeddings = []


class EmbeddingExtractor:
    """
    Embedding extractor from a layer
    Args:
        model (`nn.Module`):
            Model used for inference.
    """
    def __init__(self, model):
        self.modules = model.modules()
        self.modules_list = list(model.named_modules(remove_duplicate=False))
        self.modules_dict = dict(model.named_modules(remove_duplicate=False))
        self.emb_out = EmbeddingOutput()

    def disp_modules(self):
        """
        Display the the modules of the model.
        """
        for idx, m in enumerate(self.modules):
            if idx:
                print(idx, '->', m)

    def register(self, layer_name: str):
        """
        Registration for embedding extraction.
        Args:
            layer_name (`str`):
                Name of the layer from which the embedding is extracted.
        """
        if layer_name in self.modules_dict:
            layer_indx = 0
            for layer in self.modules:
                layer_indx = layer_indx + 1
                if self.modules_list[layer_indx][0] == layer_name:
                    _ = layer.register_forward_hook(self.emb_out)
                    break
        else:
            raise ValueError('layer_name not in modules')
