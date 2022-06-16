# Original pytorch implementation by:
# 'Bridging Video-text Retrieval with Multiple Choice Questions'
#       - https://arxiv.org/pdf/2201.04850.pdf
# Original code by / Copyright 2022, Yuying Ge.
# Modifications & additions by / Copyright 2021 Zilliz. All rights reserved.
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


def clip_initialized_model_configs():
    """
    only used in load clip initialized pretrained weights
    """
    return dict(
        embed_dim=512,
        image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
        context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12,
        is_bridge_former=True, is_bridge_former_video=True
    )
