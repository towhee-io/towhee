#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Licensed under the Apache License, Version 2.0 (the "License");
#
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
#
# modified by Zilliz.


from towhee.models.video_swin_transformer.video_swin_transformer import VideoSwinTransformer
from typing import Any, Optional, Union

import torch
from torch import nn
from torch.hub import load_state_dict_from_url



def get_all_heads(dim_in: int = 1024) -> nn.Module:
    heads = nn.ModuleDict(
        {
            "image": get_imagenet_head(dim_in),
            "rgbd": get_sunrgbd_head(dim_in),
            "video": get_kinetics_head(dim_in),
        }
    )
    return heads


def get_imagenet_head(dim_in: int = 1024) -> nn.Module:
    head = nn.Linear(in_features=dim_in, out_features=1000, bias=True)
    return head

def get_sunrgbd_head(dim_in: int = 1024) -> nn.Module:
    head = nn.Linear(in_features=dim_in, out_features=19, bias=True)
    return head

def get_kinetics_head(dim_in: int = 1024, num_classes: int = 400) -> nn.Module:
    head = nn.Linear(in_features=dim_in, out_features=num_classes, bias=True)
    return nn.Sequential(nn.Dropout(p=0.5), head)
class OmnivoreModel(nn.Module):
    """
        Args:
    """
    def __init__(self, trunk: nn.Module, heads: Union[nn.ModuleDict, nn.Module]):
        super().__init__()
        self.trunk = trunk
        self.heads = heads
        self.types = ["image", "video", "rgbd"]
        self.multimodal_model = False
        if isinstance(heads, nn.ModuleDict):
            self.multimodal_model = True
            assert [n in heads for n in self.types], "All heads must be provided"

    def head(self, features: torch.Tensor, input_type: Optional[str] = None):
        head_in = self.heads
        if self.multimodal_model:
            assert input_type in self.types, "unsupported input type"
            head_in = head_in[input_type]
        return head_in(features)

    def forward_features(self, x: torch.Tensor):
        assert x.ndim == 5
        x = self.trunk(x)
        features = [torch.mean(x, [-3, -2, -1])][0]
        return features

    def forward(self, x: torch.Tensor, input_type: Optional[str] = None):
        """
        Args:
            x: input to the model of shape 1 x C x T x H x W
            input_type: Optional[str] one of ["image", "video", "rgbd"]
                if self.multimodal_model iss True
        Returns:
            preds: tensor of shape (1, num_classes)
        """
        features = self.forward_features(x)
        return self.head(features, input_type = input_type)



CHECKPOINT_PATHS = {
    "omnivore_swinT": "https://dl.fbaipublicfiles.com/omnivore/models/swinT_checkpoint.torch",
    "omnivore_swinS": "https://dl.fbaipublicfiles.com/omnivore/models/swinS_checkpoint.torch",
    "omnivore_swinB": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_checkpoint.torch",
    "omnivore_swinB_in21k": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_In21k_checkpoint.torch",
    "omnivore_swinL_in21k": "https://dl.fbaipublicfiles.com/omnivore/models/swinL_In21k_checkpoint.torch",
    "omnivore_swinB_epic": "https://dl.fbaipublicfiles.com/omnivore/models/swinB_epic_checkpoint.torch",
}

def _omnivore_base(
    trunk: nn.Module,
    heads: Optional[Union[nn.Module, nn.ModuleDict]] = None,
    head_dim_in: int = 1024,
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    checkpoint_name: str = "omnivore_swinB",
) -> nn.Module:
    """
    Load and initialize the specified Omnivore
    model trunk (and optionally heads).
    Args:
        trunk: nn.Module of the SwinTransformer3D trunk
        heads: Provide the heads module if using a custom
            model. If not provided image/video/rgbd heads are
            added corresponding to the omnivore base model.
        head_dim_in: Only needs to be set if heads = None.
            The dim is used for the default base model heads.
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the full Omnivore model
    """
    if load_heads and heads is None:
        # Get heads
        heads = get_all_heads(dim_in=head_dim_in)

    if pretrained:
        path = CHECKPOINT_PATHS[checkpoint_name]

        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        trunk.fc_cls=nn.Sequential()
        trunk.load_state_dict(checkpoint["trunk"])

        if load_heads:
            heads.load_state_dict(checkpoint["heads"])

    if load_heads:
        model = OmnivoreModel(trunk=trunk, heads=heads)
    else:
        model = trunk

    return model

def omnivore_swinb_epic(
    progress: bool = True,
    checkpoint_name: str = "omnivore_swinB_epic",
    **kwargs: Any,
) -> nn.Module:
    r"""
    Omnivore swin B model trained on EPIC-KITCHENS-100 dataset
    Args:
        progress: print progress of loading checkpoint
    Returns:
        model: nn.Module of the omnivore model
    """

    # Only specify the non default values
    trunk = VideoSwinTransformer(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=(16, 7, 7),
        drop_path_rate=0.4,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
        **kwargs,
    )

    heads = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=3806, bias=True)
    )

    return _omnivore_base(
        trunk=trunk,
        head_dim_in=1024,  # embed_dim * 8 = 128*8
        progress=progress,
        pretrained=True,
        load_heads=True,
        checkpoint_name=checkpoint_name,
        heads=heads
    )

def omnivore_swinb(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    checkpoint_name: str = "omnivore_swinB",
    **kwargs: Any,
) -> nn.Module:
    r"""
    Omnivore model trunk: Swin B patch (2,4,4) window (1,6,7,7)
    Args:
        pretrained: if True loads weights from model trained on
            Imagenet 1k, Kinetics 400, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """

    # Only specify the non default values
    trunk = VideoSwinTransformer(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=(16, 7, 7),
        drop_path_rate=0.3,  # TODO: set this based on the final models
        patch_norm=True,  # Make this the default value?
        depth_mode="summed_rgb_d_tokens",
        **kwargs,
    )

    return _omnivore_base(
        trunk=trunk,
        head_dim_in=1024,  # embed_dim * 8 = 128*8
        progress=progress,
        pretrained=pretrained,
        load_heads=load_heads,
        checkpoint_name=checkpoint_name,
    )


def omnivore_swinb_imagenet21k(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any
) -> nn.Module:
    r"""
    Omnivore Swin B model pretrained on Imagenet 1k, Imagenet 21k,
    Kinetics 400, SUN RGBD. By default the pretrained
    weights will be loaded.
    Args:
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """

    return omnivore_swinb(
        pretrained=pretrained,
        load_heads=load_heads,
        progress=progress,
        checkpoint_name="omnivore_swinB_in21k",
        **kwargs,
    )


def omnivore_swins(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    Omnivore model trunk: Swin S patch (2,4,4) window (8,7,7)
    Args:
        pretrained: if True loads weights from model trained on
            Imagenet 1k, Kinetics 400, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """

    # Only specify the non default values
    trunk = VideoSwinTransformer(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
        **kwargs,
    )

    return _omnivore_base(
        trunk=trunk,
        head_dim_in=768,  # 96*8
        progress=progress,
        pretrained=pretrained,
        load_heads=load_heads,
        checkpoint_name="omnivore_swinS",
    )


def omnivore_swint(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    Omnivore model trunk: Swin T patch (2,4,4) window (8,7,7)
    Args:
        pretrained: if True loads weights from model trained on
            Imagenet 1k, Kinetics 400, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """

    # Only specify the non default values
    trunk = VideoSwinTransformer(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 7, 7),
        drop_path_rate=0.2,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
        **kwargs,
    )

    return _omnivore_base(
        trunk=trunk,
        head_dim_in=768,  # 96*8
        progress=progress,
        pretrained=pretrained,
        load_heads=load_heads,
        checkpoint_name="omnivore_swinT",
    )


def _omnivore_swinl(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    checkpoint_name: str = "",
    heads: Optional[nn.ModuleDict] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Omnivore model trunk: Swin L patch (2,4,4) window (8,7,7)
    Args:
        pretrained: if True loads weights from model trained on
            Imagenet 1k, Kinetics 400, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """

    assert checkpoint_name != "", "checkpoint_name must be provided"

    # Only specify the non default values
    trunk = VideoSwinTransformer(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=(8, 7, 7),
        drop_path_rate=0.3,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
        **kwargs,
    )

    return _omnivore_base(
        trunk=trunk,
        heads=heads,
        head_dim_in=1536,  # 192*8
        progress=progress,
        pretrained=pretrained,
        load_heads=load_heads,
        checkpoint_name=checkpoint_name,
    )


def omnivore_swinl_imagenet21k(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    Swin L patch 244 window 877 pretrained on Imagenet 1k, Imagenet 21k,
    Kinetics 400, SUN RGBD. By default the pretrained
    weights will be loaded.
    Args:
        pretrained: if True loads weights from model trained on
            Imagenet1k, Imagenet 21k, Kinetics 400, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """
    return _omnivore_swinl(
        pretrained=pretrained,
        progress=progress,
        load_heads=load_heads,
        checkpoint_name="omnivore_swinL_in21k",
        **kwargs,
    )


def omnivore_swinl_kinetics600(
    pretrained: bool = True,
    progress: bool = True,
    load_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    Swin L patch 244 window 877 trained with Kinetics 600
    Args:
        pretrained: if True loads weights from model trained on
            Imagenet 1k, Kinetics 600, SUN RGBD.
        progress: print progress of loading checkpoint
        load_heads: if True, loads the 3 heads, one each for
            image/video/rgbd prediction. If False loads only the
            trunk.
    Returns:
        model: nn.Module of the omnivore model
    """

    heads = nn.ModuleDict(
        {
            "image": get_imagenet_head(dim_in=1536),
            "rgbd": get_sunrgbd_head(dim_in=1536),
            "video": get_kinetics_head(dim_in=1536, num_classes=600),
        }
    )

    return _omnivore_swinl(
        pretrained=pretrained,
        progress=progress,
        load_heads=load_heads,
        checkpoint_name="omnivore_swinL_kinetics600",
        heads=heads,
        **kwargs,
    )

def create_model(
        model_name: str = None,
        pretrained: bool = True,
        device: str = "cpu",
        ):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "omnivore_swinT":
        model = omnivore_swint(pretrained = pretrained)
    elif model_name == "omnivore_swinS":
        model = omnivore_swins(pretrained = pretrained)
    elif model_name == "omnivore_swinB":
        model = omnivore_swinb(pretrained = pretrained)
    elif model_name == "omnivore_swinB_in21k":
        model = omnivore_swinb_imagenet21k(pretrained = pretrained)
    elif model_name == "omnivore_swinL_in21k":
        model = omnivore_swinl_imagenet21k(pretrained = pretrained)
    elif model_name == "omnivore_swinB_epic":
        model = omnivore_swinb_epic(pretrained = pretrained)
    else:
        raise AttributeError(f"Invalid model_name {model_name}.")
    model.to(device)
    return model
