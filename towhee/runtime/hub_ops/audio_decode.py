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


class AudioDecode:
    """
    `Audio Decode <https://towhee.io/audio-decode/ffmpeg>`_ converts the encoded audio back to uncompressed audio frames.
    In most cases, audio decoding is the first step of an audio processing pipeline.
    """

    ffmpeg: HubOp = HubOp('audio_decode.ffmpeg')
    """
    Decode audio by ffmpeg lib.

    __init__(self, batch_size=-1, sample_rate=None, layout=None) -> None:
        batch_size(`int`):
            if set, __call__ function return Generator[List[AudioFrame]], else return Generator[AudioFrame]
        sample_rate(`int`):
            The target sample rate.
        layout(`str`):
            The target layout, e.g. 'stereo', 'mono'

    __call__(self, audio_path: str) -> Generator[Union[towhee.types.AudioFrame, List[towhee.types.AudioFrame.]]]
        audio_path(`str`):
            Audio file path.
        Return(`Generator[Union[towhee.types.AudioFrame, List[towhee.types.AudioFrame.]]]`):
            An generator over audio frames with type towhee.types.AudioFrame.

    Example;

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('audio_file')
            .flat_map('audio_file', 'frame', ops.audio_decode.ffmpeg())
            .output('frame')
        )

        DataCollection(p('./music.mp3')).show(limit=1)
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return HubOp('towhee.audio_decode')(*args, **kwds)
