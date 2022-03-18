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


class DataSourceMixin:
    """
    Mixin for loading data from different data sources
    """

    # pylint: disable=import-outside-toplevel
    @classmethod
    def from_glob(cls, pattern):
        """
        generate a file list with `pattern`
        """
        from glob import glob
        return cls.stream(glob(pattern))

    @classmethod
    def from_zip(cls, url, pattern, mode=None):
        """load files from url/path.

        Args:
            zip_src (`Union[str, path]`):
                The path leads to the image.
            pattern (`str`):
                The filename pattern to extract.
            mode (str):
                file open mode.

        Returns:
            (File): The file handler for file in the zip file.
        """
        from towhee.utils.repo_normalize import RepoNormalize
        from io import BytesIO
        from zipfile import ZipFile
        from pathlib import Path
        from glob import glob

        from urllib.request import urlopen

        def inner():
            if RepoNormalize(str(url)).url_valid():
                with urlopen(url) as zip_file:
                    zip_path = BytesIO(zip_file.read())
            else:
                zip_path = str(Path(url).resolve())
            with ZipFile(zip_path, 'r') as zfile:
                file_list = zfile.namelist()
                path_list = glob.fnmatch.filter(file_list, pattern)
                for path in path_list:
                    with zfile.open(path, mode=mode) as f:
                        yield f
        return cls.stream(inner())

    @classmethod
    def from_camera(cls, device_id=0, limit=-1):
        """
        read images from a camera.
        """
        import cv2
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
