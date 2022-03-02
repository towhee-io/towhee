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

from glob import glob


class DataSourceMixin:
    """
    Mixin for loading data from different data sources
    """

    @classmethod
    def from_glob(cls, pattern):
        """
        generate a file list with `pattern`
        """
        return cls.stream(glob(pattern))

    @classmethod
    def from_zip(cls, zip_path, pattern):
        from towhee.utils.ndarray_utils import from_zip # pylint: disable=import-outside-toplevel
        return cls.stream(from_zip(zip_path, pattern))

    @classmethod
    def from_camera(cls, device_id=0, limit=-1):
        """
        read images from a camera.
        """
        import cv2 # pylint: disable=import-outside-toplevel
        cnt = limit
        def inner():
            nonlocal cnt
            cap = cv2.VideoCapture(device_id)
            while cnt != 0:
                retval, im = cap.read()
                if retval:
                    yield im
                    cnt -= 1
        return cls.stream(inner())
