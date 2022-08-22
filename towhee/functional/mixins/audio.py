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

# pylint: disable=import-outside-toplevel
from towhee.types.audio_frame import AudioFrame


class AudioMixin():
    """
    Mixin for audio processing.
    """
    @classmethod
    def read_audio(cls, path): # pragma: no cover
        from towhee.utils.thirdparty.av_utils import av

        def inner():
            acontainer = av.open(path)
            stream = acontainer.streams.audio[0]

            for frame in acontainer.decode(stream):
                timestamp = int(frame.time * 1000)
                sample_rate = frame.sample_rate
                layout = frame.layout.name
                ndarray = frame.to_ndarray()
                yield AudioFrame(ndarray, sample_rate, timestamp, layout)

        return cls(inner())
