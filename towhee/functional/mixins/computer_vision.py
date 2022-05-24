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
    def read_video(cls, path):

        def inner():
            from towhee.utils.cv2_utils import cv2

            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    yield frame
                else:
                    cap.release()

        return cls(inner())

    def to_video(self, path):
        from towhee.utils.cv2_utils import cv2

        out = None
        for frame in self:
            if out is None:
                out = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                    (frame.shape[1], frame.shape[0]))
            out.write(frame)
