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
import pprint

class EmbeddingOutput:
    """
    Container for embedding extractor.
    """
    def __init__(self):
        self.embeddings = []

    def __call__(self, module, module_in, module_out):
        self.embeddings.append(module_out)

    def clear(self):
        """
        clear list
        """
        self.embeddings = []


class EmbeddingExtractor:
    """
    Embedding extractor from a layer
    Args:
        model (`nn.Module`):
            Model used for inference.
    """
    def __init__(self, model):
        # self.modules = model.modules()
        # self.modules_list = list(model.named_modules(remove_duplicate=False))
        self.modules_dict = dict(model.named_modules(remove_duplicate=False))
        self.emb_out = EmbeddingOutput()

    def disp_modules(self, full=False):
        """
        Display the the modules of the model.
        """
        if not full:
            pprint.pprint(list(self.modules_dict.keys()))
        else:
            pprint.pprint(self.modules_dict)

    def register(self, layer_name: str):
        """
        Registration for embedding extraction.
        Args:
            layer_name (`str`):
                Name of the layer from which the embedding is extracted.
        """
        if layer_name in self.modules_dict:
            layer = self.modules_dict[layer_name]
            layer.register_forward_hook(self.emb_out)
        else:
            raise ValueError('layer_name not in modules')
