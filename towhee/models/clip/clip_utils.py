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

import os
import hashlib
import urllib.request
from tqdm import tqdm
from typing import Union, List
from pkg_resources import packaging
import warnings

import torch

from .simple_tokenizer import SimpleTokenizer


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading file.")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")),
                  ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    with open(download_target, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def patch_device(module, device_node):
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


def patch_float(module, float_node):
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


def base_configs():
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


def get_configs(model_name):
    if model_name == "clip_resnet_r50":
        configs = base_configs()
        configs.update(dict(
            url="https://openaipublic.azureedge.net/clip/models/ \
            afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
            embed_dim=1024,
            vision_layers=(3, 4, 6, 3),
            vision_width=64,
            vision_patch_size=None,
        ))
    elif model_name == "clip_resnet_r101":
        configs = base_configs()
        configs.update(dict(
            url="https://openaipublic.azureedge.net/clip/models/ \
            8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
            embed_dim=512,
            vision_layers=(3, 4, 23, 3),
            vision_width=64,
            vision_patch_size=None,
        ))
    elif model_name == "clip_vit_b16":
        configs = base_configs()
        configs.update(dict(
            url="https://openaipublic.azureedge.net/clip/models/\
            5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        ))
    elif model_name == "clip_vit_b32":
        configs = base_configs()
        configs.update(dict(
            url=("https://openaipublic.azureedge.net/clip/models/"
                 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
            vision_patch_size=32
        ))
    else:
        raise ValueError(f"Invalid model name '{model_name}'.")
    return configs
