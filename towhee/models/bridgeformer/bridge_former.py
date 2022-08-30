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
from towhee.models import clip
from towhee.models.frozen_in_time import FrozenInTime
from towhee.models.bridgeformer.bridge_former_training import BridgeFormerTraining
from towhee.models.frozen_in_time.frozen_utils import state_dict_data_parallel_fix
from towhee.models.clip.clip_utils import convert_weights
import torch


def create_model(
        model_name: str = None,
        pretrained: bool = False,
        weights_path: str = None,
        device: str = None,
        **kwargs
):
    """
    Create BridgeFormer model.

    Args:
        model_name (`str`):
            - "clip_initialized_model": use the clip_vit_b32 model.
            - "bridge_former_training": use the bridge former training model Architecture.
            - Others: use the default inference bridge former model Architecture.
        pretrained (`bool`):
            Flag to control if load pretrained model.
        weights_path (`str`):
            The path of local model weights.
        device (`str`):
            Device, defaults to None. If None, use cuda when it is available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained and weights_path is None:
        raise AssertionError("if pretrained is true, weights_path needs to be specified")

    if model_name == "clip_initialized_model":
        model = clip.create_model(model_name="clip_vit_b32", pretrained=False,
                                  is_bridge_former=True, is_bridge_former_video=True, **kwargs)
        if pretrained:
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = checkpoint["state_dict"]
            state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in state_dict:
                    del state_dict[key]
            if device != "cpu":
                # CPU training, does not support half-precision numbers
                convert_weights(model)
            model.load_state_dict(state_dict, strict=True)

    elif model_name == "bridge_former_training":
        model = BridgeFormerTraining(weights_path=weights_path,
                                     is_pretrained=pretrained,
                                     device=device,
                                     **kwargs)

    else:
        model = FrozenInTime(weights_path=weights_path,
                             is_pretrained=pretrained,
                             attention_style="bridge_former",
                             device=device,
                             **kwargs)

    model.eval()
    model.to(device)
    return model

