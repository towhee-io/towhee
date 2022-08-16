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


class ComputerVisionMixin:
    """
    Mixin for computer vision problems.
    """

    def image_imshow(self, title='image'):  # pragma: no cover
        from towhee.utils.cv2_utils import cv2
        for im in self:
            cv2.imshow(title, im)
            cv2.waitKey(1)

    @classmethod
    def read_camera(cls, device_id=0, limit=-1):  # pragma: no cover
        """
        read images from a camera.
        """
        from towhee.utils.cv2_utils import cv2
        cnt = limit

        def inner():
            nonlocal cnt
            cap = cv2.VideoCapture(device_id)
            while cnt != 0:
                retval, im = cap.read()
                if retval:
                    yield im
                    cnt -= 1

        return cls(inner())

    # pylint: disable=redefined-builtin
    @classmethod
    def read_video(cls, path, format='rgb24'):
        """
        Load video as a datacollection.

        Args:
            path:
                The path to the target video.
            format:
                The format of the images loaded from video.
        """
        from towhee.utils.av_utils import av

        vcontainer = av.open(path)
        video_stream = vcontainer.streams.video[0]

        frames = vcontainer.decode(video_stream)
        images = (frame.to_rgb().to_ndarray(format=format) for frame in frames)

        return cls(images)

    def to_video(self,
                 output_path,
                 codec=None,
                 rate=None,
                 width=None,
                 height=None,
                 format=None,
                 template=None,
                 audio_src=None):
        """
        Encode a video with audio if provided.

        Args:
            output_path:
                The path of the output video.
            codec:
                The codec to encode and decode the video.
            rate:
                The rate of the video.
            width:
                The width of the video.
            height:
                The height of the video.
            format:
                The format of the video frame image.
            template:
                The template video stream of the ouput video stream.
            audio_src:
                The audio to encode with the video.
        """
        from towhee.utils.av_utils import av
        import itertools

        output_container = av.open(output_path, 'w')
        codec = codec if codec else template.name if isinstance(
            template, av.video.stream.VideoStream) else None
        rate = rate if rate else template.average_rate if isinstance(
            template, av.video.stream.VideoStream) else None
        width = width if width else template.width if isinstance(
            template, av.video.stream.VideoStream) else None
        height = height if height else template.height if isinstance(
            template, av.video.stream.VideoStream) else None
        format = format if format else 'rgb24'

        output_video = None
        output_audio = None

        if audio_src:
            acontainer = av.open(audio_src)
            audio_stream = acontainer.streams.audio[0]
            output_audio = output_container.add_stream(
                codec_name=audio_stream.name, rate=audio_stream.rate)
            for aframe, array in itertools.zip_longest(
                    acontainer.decode(audio_stream), self):
                if array is not None:
                    if not output_video:
                        height = height if height else array.shape[0]
                        width = width if width else array.shape[1]
                        output_video = output_container.add_stream(
                            codec_name=codec,
                            rate=rate,
                            width=width,
                            height=height)
                    vframe = av.VideoFrame.from_ndarray(array, format=format)
                    vpacket = output_video.encode(vframe)
                    output_container.mux(vpacket)
                if aframe:
                    apacket = output_audio.encode(aframe)
                    output_container.mux(apacket)
        else:
            for array in self:
                if not output_video:
                    height = height if height else array.shape[0]
                    width = width if width else array.shape[1]
                    output_video = output_container.add_stream(
                        codec_name=codec,
                        rate=rate,
                        width=width,
                        height=height)
                vframe = av.VideoFrame.from_ndarray(array, format=format)
                vpacket = output_video.encode(vframe)
                output_container.mux(vpacket)

        for vpacket in output_video.encode():
            output_container.mux(vpacket)

        if output_audio:
            for apacket in output_audio.encode():
                output_container.mux(apacket)

        output_container.close()
