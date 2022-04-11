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
        model_name (`str`):
            model name
    Returns:
        A dictionary including tensors for both video and audio.

    Example:
        >>> from towhee.models.utils.video_transforms import VideoTransforms
        >>> tfms = VideoTransforms("slowfast_r50")
        >>> video_data = tfms(video_path="path/to/video", start_sec=0, end_sec=30)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tfms_params = video_configs[model_name]
        self.tfms = ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(self.tfms_params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(
                    mean=self.tfms_params["mean"],
                    std=self.tfms_params["std"],
                    inplace=True
                ),
                ShortSideScale(size=self.tfms_params["side_size"]),
                CenterCropVideo(
                    size=(self.tfms_params["crop_size"], self.tfms_params["crop_size"])
                ),
                CollectFrames(),
                PackPathway(alpha=self.tfms_params["alpha"]) if model_name.startswith("slowfast") else nn.Identity()
            ]),
        )

    def __call__(self, video_path: str, start_sec: float = 0.0, end_sec: float = 30.0):
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = self.tfms(video_data)
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
        "sampling_rate": 8,
        "mean": [0.45, 0.45, 0.45],
        "std": [0.225, 0.225, 0.225],
        }
    configs.update(**kwargs)
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
}
