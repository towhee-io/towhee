# Inspired by pytorchvideo / Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Inspired by torchvision: https://github.com/pytorch/vision
# Modifications by Copyright 2022 Zilliz . All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging

import numpy
import numbers
import torch
from torch import nn

from torchvision.transforms import Compose

try:
    from pytorchvideo.transforms import (
        ShortSideScale,
        UniformTemporalSubsample,
        # UniformCropVideo
    )
except ModuleNotFoundError:
    os.system('pip install "git+https://github.com/facebookresearch/pytorchvideo.git"')
    from pytorchvideo.transforms import (
        ShortSideScale,
        UniformTemporalSubsample,
        # UniformCropVideo
    )

log = logging.getLogger()


def transform_video(
        video: numpy.ndarray,
        model_name: str = None,
        **kwargs):
    if model_name:
        cfg = video_configs[model_name]
        cfg.update(model_name=model_name, **kwargs)
    else:
        cfg = get_configs(**kwargs)
    tsfm = VideoTransforms(cfg)
    output = tsfm(video)
    return output


class VideoTransforms:
    """
    Transform video to video & audio tensors given an video path, corresponding to models by model name.
    The code is buiil on top of examples given by [Pytorchvideo](https://pytorchvideo.org/).
    The following model names are supported:
        - slow_r50
        - slowfast_r50
        - slowfast_r101
        - x3d_xs
        - x3d_s
        - x3d_m

    Args:
        cfg:
            configs including parameters
    Returns:
        A dictionary including tensors for video (and audio).

    Example:
        >>> from towhee.models.utils.video_transforms import VideoTransforms
        >>>
        >>> model_name = "x3d_xs"
        >>> cfg = video_configs[model_name]
        >>> cfg.update(model_name=model_name)
        >>> tsfm = VideoTransforms(cfg)
        >>> output = tsfm(video)
        >>> print(output.shape)
        torch.Size([3, 4, 182, 182])
    """

    def __init__(self, cfg):
        if "model_name" in cfg.keys():
            self.model_name = cfg["model_name"]
        else:
            self.model_name = "default"
        if "sampling_rate" in cfg.keys():
            self.sampling_rate = cfg["sampling_rate"]
        else:
            self.sampling_rate = None
        if "alpha" in cfg.keys():
            self.alpha = cfg["alpha"]
        else:
            self.alpha = None
        try:
            self.num_frames = cfg["num_frames"]
            self.mean = cfg["mean"]
            self.std = cfg["std"]
            self.side_size = cfg["side_size"]
            self.crop_size = cfg["crop_size"]
        except KeyError as e:
            log.error("Invalid key in configs: %s", e)
            raise KeyError from e

        tfms_list = [UniformTemporalSubsample(self.num_frames),
                     NormalizeVideo(mean=self.mean, std=self.std, inplace=True),
                     ShortSideScale(size=self.side_size),
                     CenterCropVideo(crop_size=self.crop_size),
                     PackPathway(alpha=self.alpha)]
        if self.num_frames is None:
            del tfms_list[0]
        if not self.model_name.startswith("slowfast"):
            del tfms_list[-1]
        self.tfms = Compose(tfms_list)
        if self.model_name.startswith("slowfast"):
            log.info("Using PackPathway for slowfast model.")

    # video: shape (c, t, w, h)
    def __call__(self, video: numpy.ndarray):
        assert video.dtype == numpy.float32
        video = torch.from_numpy(video)

        video_data = self.tfms(video)
        return video_data


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors.

    Args:
        alpha (`int`):
            alpha value

    Returns:
        a list of tensors [slow_pathway, fast_pathway]
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class CenterCropVideo:
    """
    Original code from torchvision: https://github.com/pytorch/vision/tree/main/torchvision/transforms

    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    Returns:
        torch.tensor: central cropping of video clip. Size is
        (C, T, crop_size, crop_size)
    """
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        assert clip.ndimension() == 4
        h, w = clip.size(-2), clip.size(-1)
        th, tw = self.crop_size
        if h < th or w < tw:
            raise ValueError("height and width must be no smaller than crop_size")

        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return clip[..., i: i + th, j: j + tw]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"


class NormalizeVideo:
    """
    Original code from torchvision: https://github.com/pytorch/vision/tree/main/torchvision/transforms

    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        assert clip.ndimension() == 4
        if not self.inplace:
            clip = clip.clone()
        mean = torch.as_tensor(self.mean, dtype=clip.dtype, device=clip.device)
        std = torch.as_tensor(self.std, dtype=clip.dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


def get_configs(**kwargs):
    configs = {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 8,
        "mean": [0.45, 0.45, 0.45],
        "std": [0.225, 0.225, 0.225],
        }
    configs.update(kwargs)
    return configs


video_configs = {
    "slow_r50": get_configs(),
    "c2d_r50": get_configs(),
    "i3d_r50": get_configs(),
    "slowfast_r50": get_configs(
        num_frames=32,
        sampling_rate=2,
        alpha=4
        ),
    "slowfast_r101": get_configs(
        num_frames=32,
        sampling_rate=8,
        alpha=4
        ),
    "x3d_xs": get_configs(
        side_size=182,
        crop_size=182,
        num_frames=4,
        sampling_rate=12
        ),
    "x3d_s": get_configs(
        side_size=182,
        crop_size=182,
        num_frames=13,
        sampling_rate=6
        ),
    "x3d_m": get_configs(
        num_frames=16,
        sampling_rate=5
        ),
    "mvit_base_16x4": get_configs(
        side_size=224,
        crop_size=224,
        num_frames=16,
        sampling_rate=4
        ),
    "mvit_base_32x3": get_configs(
        side_size=224,
        crop_size=224,
        num_frames=32,
        sampling_rate=3
        ),
    "csn_r101": get_configs(
        num_frames=32,
        sampling_rate=2
        ),
    "r2plus1d_r50": get_configs(
        num_frames=16,
        sampling_rate=4
        )
}
