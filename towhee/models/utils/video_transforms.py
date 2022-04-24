# Original implementation by Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
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
from typing import Union

import numpy
import torch
from torch import nn

from torchvision.transforms import Compose, Lambda

try:
    from torchvideo.transforms import (
        CenterCropVideo,
        NormalizeVideo,
        CollectFrames,
        # PILVideoToTensor
    )
except ModuleNotFoundError:
    os.system('pip install "git+https://github.com/willprice/torchvideo.git"')
    from torchvideo.transforms import (
        CenterCropVideo,
        NormalizeVideo,
        CollectFrames,
        # PILVideoToTensor
    )

try:
    from pytorchvideo.data.encoded_video import EncodedVideo
except ModuleNotFoundError:
    os.system('pip install "git+https://github.com/facebookresearch/pytorchvideo.git"')
    from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    # UniformCropVideo
)

log = logging.getLogger()


def transform_video(
        video: Union[str, numpy.ndarray],
        model_name: str = None,
        start_sec: float = 0.,
        end_sec: float = 30.,
        **kwargs):
    if model_name:
        cfg = video_configs[model_name]
        cfg.update(model_name=model_name)
    else:
        cfg = get_configs(**kwargs)
    tsfm = VideoTransforms(cfg)
    output = tsfm(video=video, start_sec=start_sec, end_sec=end_sec)
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
        try:
            self.num_frames = cfg["num_frames"]
            self.mean = cfg["mean"]
            self.std = cfg["std"]
            self.side_size = cfg["side_size"]
            self.crop_size = cfg["crop_size"]
        except KeyError as e:
            log.error("Invalid key in configs: %s", e)
            raise KeyError from e

        self.tfms = ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(self.num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(
                    mean=self.mean,
                    std=self.std,
                    inplace=True
                ),
                ShortSideScale(size=self.side_size),
                CenterCropVideo(
                    size=(self.crop_size, self.crop_size)
                ),
                CollectFrames(),
                PackPathway(alpha=self.alpha) if self.model_name.startswith("slowfast") else nn.Identity()
                ]),
            )
        if self.model_name.startswith("slowfast"):
            log.info("Using PackPathway for slowfast model.")

    def __call__(self, video: Union[str, numpy.ndarray], start_sec: float = 0.0, end_sec: float = 30):
        if isinstance(video, str):
            video = EncodedVideo.from_path(video)
            video = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            if self.sampling_rate:
                total_frames = self.num_frames * self.sampling_rate
                frames_per_sec = total_frames / (end_sec - start_sec)
                log.info("Frames per second: %s", frames_per_sec)
        elif isinstance(video, numpy.ndarray):
            assert video.dtype == numpy.float32
            video = dict(video=torch.from_numpy(video))

        video_data = self.tfms(video)
        if isinstance(video_data["video"], list) and str(self.tfms._transform.transforms[-1]) == "PackPathway()":
            video_data["video_slow"] = video_data["video"][0]
            video_data["video_fast"] = video_data["video"][1]
        else:
            video_data["video"] = torch.stack(video_data["video"])
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
        frames = torch.stack(frames)
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
