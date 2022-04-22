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


from typing import NamedTuple, List


from towhee._types import Image
from towhee.operator import Operator, SharedType


Cv2Outputs = NamedTuple('Outputs', [('img', 'Image')])


class Cv2Decoder(Operator):
    """
    Cv2 decoder
    """

    def __init__(self, count=10):
        super().__init__()
        self.count = count

    def __call__(self, video_path: str):
        import cv2
        import time
        cap = cv2.VideoCapture(video_path)
        while True:
            if self.count <= 0:
                break
            _, frame = cap.read()
            if frame is not None:
                yield Cv2Outputs(Image(frame, 'BGR'))
                self.count -= 1
                time.sleep(0.1)
            else:
                cap.release()
                break

    @property
    def shared_type(self):
        return SharedType.NotReusable
