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
import torch
from torch import nn

from towhee.models.multiscale_vision_transformers.mvit import MViT


def create_mvit_model(
        model_name: str = "imagenet_b_16_conv",
        checkpoint_path: str = None,
        device: str = None,
        change_model_keys: bool = True
) -> nn.Module:
    """
    Create Multiscale Vision Transformers model.
    https://arxiv.org/abs/2104.11227

    Args:
        model_name (`str`):
            Multiscale Vision Transformers model name.
        checkpoint_path (`str`):
            Local checkpoint path, default is None.
            Checkpoint weights can be download in https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md.
        device (`str`):
            Model device, cpu or cuda.
        change_model_keys (`bool`):
            This MViT structure is a little different from that from Facebookresearch for visualization.
            So you should set `change_model_keys` is True if you download pretrained checkpoint from Facebookresearch.

    """

    def _change_model_keys(model_state):
        for key in list(model_state.keys()):
            if "attn.pool" in key or "attn.norm" in key:
                key_word_list = key.split(".")
                kqv = key_word_list[3][-1]
                norm_pool = key_word_list[3][:4]
                new_key = ".".join(
                    [key_word_list[0], key_word_list[1], "attn", "atn_pool_" + kqv, norm_pool,
                     key_word_list[-1]])
                print(f"old_key: {key}\t new_key: {new_key}")
                model_state[new_key] = model_state.pop(key)
        return model_state

    config = {}
    if model_name == "imagenet_b_16_conv":
        config = {
            "patch_2d": True,
            "patch_stride": [4, 4],
            "embed_dim": 96,
            "num_heads": 1,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "dropout_rate": 0.0,
            "depth": 16,
            "droppath_rate": 0.1,
            "mode": "conv",
            "cls_embed_on": True,
            "sep_pos_embed": False,
            "norm": "layernorm",
            "patch_kernel": [7, 7],
            "patch_padding": [3, 3],
            "pool_q_kernel": [[], [1, 3, 3], [], [1, 3, 3], [], [], [], [], [], [], [], [], [], [], [1, 3, 3], []],
            "pool_kv_kernel": [[1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3],
                               [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
            "pool_skip_kernel": [[], [1, 3, 3], [], [1, 3, 3], [], [], [], [], [], [], [], [], [], [], [1, 3, 3], []],
            "pool_q_stride": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
            "pool_kv_stride": [[1, 4, 4], [1, 2, 2], [1, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                               [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "pool_skip_stride": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
            "dim_mul_arg": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "head_mul_arg": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "norm_stem": False,
            "num_classes": 1000,
            "head_act": "softmax",
            "train_crop_size": 224,
            "test_crop_size": 224,
            "num_frames": 1,
            "input_channel_num": [3],
        }

    model = MViT(**config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint["model_state"]
        if change_model_keys:
            model_state = _change_model_keys(model_state)
        model.load_state_dict(model_state)
    return model
