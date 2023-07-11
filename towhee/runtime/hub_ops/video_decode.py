# Copyright 2023 Zilliz. All rights reserved.
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


from typing import Any
from towhee.runtime.factory import HubOp


class VideoDecode:
    """
    Video deocde, in most cases, video decode is the first step of an video processing pipeline.
    """
    ffmpeg: HubOp = HubOp('video_decode.ffmpeg')
    """
    Decode videos by ffmpeg lib.

    __init__(self, start_time=None, end_time=None, sample_type=None, args=None)
        start_time(`float`):
            Decode the video starting from start_time, defaulting to decoding from the beginning.
        end_time(`float`):
            Decode until end_time, defaulting to decoding to the end.
        sample_type(`str`):
            Sampling type, `uniform_temporal_subsample` or `time_step_sample`, no sampling by default。
        args(`dict`):
            if sample_type is `uniform_temporal_subsample`:
                num_samples: int
                  total number of frames to sample.
            if sample_type is `time_step_sample`:
                time_step: int
                    time interval between samples.

    __call__(self, video_path: str) -> Generator[towhee.types.VideoFrame]
        video_path(`str`)
            support local path and http/https url.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('video_file')
            .flat_map('video_file', 'frame',
                      ops.video_decode.ffmpeg(start_time=10.0, end_time=15.0, sample_type='time_step_sample', args={'time_step': 1})
            )
            .output('frame')
        )

        DataCollection(p('./video.mp4')).show(limit=1)
    """

    VPF: HubOp = HubOp('video_decode.VPF')
    """
    `GPU video decoder <https://towhee.io/video-decode/VPF>`_, using https://github.com/NVIDIA/VideoProcessingFramework,
    Users need to install the vpf package by themselves. GPC decode only support h.264, h.265 and vp9, others will use cpu decode.
    4% diff with cpu-decode.

    __init__(self, gpu_id, start_time=None, end_time=None, sample_type=None, args=None)
        gpu_id(`int`):
            GPU id
        start_time(`float`):
            Decode the video starting from start_time, defaulting to decoding from the beginning.
        end_time(`float`):
            Decode until end_time, defaulting to decoding to the end.
        sample_type(`str`):
            Sampling type, `uniform_temporal_subsample` or `time_step_sample`, no sampling by default。
        args(`dict`):
            if sample_type is `uniform_temporal_subsample`
                num_samples: int
                    total number of frames to sample.
            if sample_type is `time_step_sample`
                time_step: int
                    time interval between samples.

    __call__(self, video_path: str) -> Generator[towhee.types.VideoFrame]
        video_path(`str`)
            support local path and http/https url.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('video_file')
            .flat_map('video_file', 'frame',
                      ops.video_decode.VPF(start_time=10.0, end_time=15.0, sample_type='time_step_sample', args={'time_step': 1})
            )
            .output('frame')
        )

        DataCollection(p('./video.mp4')).show(limit=1)
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return HubOp('towhee.video_decode')(*args, **kwds)

