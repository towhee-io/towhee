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

from typing import Union
from pathlib import Path

from towhee.functional.entity import Entity


class DatasetMixin:
    """
    Mixin for dealing with dataset
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
    def from_zip(cls, url, pattern, mode='r'):
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
        from glob import fnmatch

        from urllib.request import urlopen

        def inner():
            if RepoNormalize(str(url)).url_valid():
                with urlopen(url) as zip_file:
                    zip_path = BytesIO(zip_file.read())
            else:
                zip_path = str(Path(url).resolve())
            with ZipFile(zip_path, 'r') as zfile:
                file_list = zfile.namelist()
                path_list = fnmatch.filter(file_list, pattern)
                for path in path_list:
                    with zfile.open(path, mode=mode) as f:
                        yield f.read()

        return cls.stream(inner())

    @classmethod
    def from_camera(cls, device_id=0, limit=-1):
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

        return cls.stream(inner())

    # def image_decode(self):
    #     pass

    # def image_encode(self):
    #     pass

    def audio_decode(self):
        pass

    def audio_encode(self):
        pass

    def video_decode(self):
        pass

    def video_encode(self):
        pass

    @classmethod
    def from_json(cls, json_path: Union[str, Path], encoding: str = 'utf-8'):
        import json

        def inner():
            with open(json_path, 'r', encoding=encoding) as f:
                string = f.readline()
                while string:
                    data = json.loads(string)
                    string = f.readline()
                    yield Entity(**data)

        return cls.stream(inner())

    @classmethod
    def from_csv(cls, csv_path: Union[str, Path], encoding: str = 'utf-8-sig'):
        import csv

        def inner():
            with open(csv_path, 'r', encoding=encoding) as f:
                data = csv.DictReader(f)
                for line in data:
                    yield Entity(**line)

        return cls.stream(inner())

    def random_sample(self):
        # core API already exists
        pass

    def filter_data(self):
        # core API already exists
        pass

    # pylint: disable=dangerous-default-value
    def split_train_test(self, size: list = [0.9, 0.1], **kws):
        """
        Split DataCollection to train and test data.

        Args:
            size (`list`):
                The size of the train and test.

        Examples:

        >>> from towhee.functional import DataCollection
        >>> dc = DataCollection.range(10)
        >>> train, test = dc.split_train_test(shuffle=False)
        >>> train.to_list()
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> test.to_list()
        [9]
        """
        from towhee.utils import sklearn_utils
        train_size = size[0]
        test_size = size[1]
        train, test = sklearn_utils.train_test_split(list(self), train_size=train_size, test_size=test_size, **kws)
        return self._factory(train), self._factory(test)

    def save_json(self):
        pass

    def save_image(self):
        pass

    def save_csv(self):
        pass

    def get_info(self):
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
