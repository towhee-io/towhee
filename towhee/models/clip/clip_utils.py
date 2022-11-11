# Built on top of the original implementation at https://github.com/openai/CLIP
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

from typing import Union, List, Dict
from pkg_resources import packaging
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

import torch
import torchvision
from torch import nn

from .simple_tokenizer import SimpleTokenizer

BICUBIC = InterpolationMode.BICUBIC

def base_configs() -> Dict:
    return dict(
        # vision
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        # text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )


def get_configs(model_name: str) -> Dict:
    """
    Given a clip model name, return the default configs.
    Args:
        model_name (`str`):
            Clip model name.

    Returns:
        (`Dict`):
            Default configs for given model name.
    """
    if model_name == "clip_resnet_r50":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
            embed_dim=1024,
            vision_layers=(3, 4, 6, 3),
            vision_width=64,
            vision_patch_size=None,
        ))
    elif model_name == "clip_resnet_r101":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
            embed_dim=512,
            vision_layers=(3, 4, 23, 3),
            vision_width=64,
            vision_patch_size=None,
        ))
    elif model_name == "clip_vit_b16":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
        ))
    elif model_name == "clip_vit_b32":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
            vision_patch_size=32,
            multilingual_model="M-CLIP/XLM-Roberta-Large-Vit-B-32"
        ))
    elif model_name == "clip_resnet_r50x4":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"),
        ))
    elif model_name == "clip_resnet_r50x16":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt"),
        ))
    elif model_name == "clip_resnet_r50x64":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt"),
        ))
    elif model_name == "clip_vit_l14":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
        ))
    elif model_name == "clip_vit_l14@336px":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
        ))
    else:
        raise ValueError(f"Invalid model name '{model_name}'.")
    return configs

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_transforms(model_name: str) -> torchvision.transforms.Compose:
    pixels = {}
    pixels["clip_resnet_r50"] = 224
    pixels["clip_resnet_r101"] = 224
    pixels["clip_vit_b16"] = 224
    pixels["clip_vit_b32"] = 224
    pixels["clip_resnet_r50x4"] = 288
    pixels["clip_resnet_r50x16"] = 384
    pixels["clip_resnet_r50x64"] = 448
    pixels["clip_vit_l14"] = 224
    pixels["clip_vit_l14@336px"] = 336
    return _transform(pixels[model_name])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) \
        -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters

    Args:
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

    Returns:
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = SimpleTokenizer()

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def patch_device(model: nn.Module, device: str):
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_dev(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_dev)
    patch_dev(model.encode_image)
    patch_dev(model.encode_text)


def patch_float(model: nn.Module):
    float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
    float_node = float_input.node()

    def patch_flt(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("aten::to"):
                inputs = list(node.inputs())
                for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                    if inputs[i].node()["value"] == 5:
                        inputs[i].node().copyAttributes(float_node)

    model.apply(patch_flt)
    patch_flt(model.encode_image)
    patch_flt(model.encode_text)

    model.float()
