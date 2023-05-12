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

import numpy
from tabulate import tabulate

from towhee.types import Image, VideoFrame, AudioFrame

class NestedConsoleTable:
    """Convert nested  data structure to tabulate

       data example:
       {
            "headers": ['Node', 'Inputs', 'Outputs'],
            "data": [{
                "headers": ['a', 'b'],
                 "data": [1, 2]
             },{
                 "headers": ['a', 'b'],
                 "data": [{...}]
             }]
        }
    """
    def __init__(self, data):
        self._data = data

    @staticmethod
    def to_tabulate(data, headers):
        """Convert two dimensional data structure into tabulate

        Args:
            data (List)
            headers (Iterable[str]): The names of columns defined by user.

        Returns:
            str: tabulate str
        """
        tb_contents = []
        for r in data:
            cells = []
            for x in r:
                if isinstance(x, dict) and 'headers' in x and 'data' in x:
                    cells.append(NestedConsoleTable.to_tabulate(x['data'], x['headers']))
                else:
                    cells.append(_to_plain_cell(x))
            tb_contents.append(cells)
        return tabulate(tb_contents, headers=headers, tablefmt='grid')

    def show(self):
        table = NestedConsoleTable.to_tabulate(self._data['data'], self._data['headers'])
        print(table)


def _to_plain_cell(data):  # pragma: no cover
    if isinstance(data, str):
        return _text_brief(data)
    if isinstance(data, (Image, VideoFrame)):
        return _image_brief(data)
    elif isinstance(data, AudioFrame):
        return _audio_frame_brief(data)
    elif isinstance(data, numpy.ndarray):
        return _ndarray_brief(data)

    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return _list_brief(data, _text_brief)
        elif all(isinstance(x, (Image, VideoFrame)) for x in data):
            return _list_brief(data, _image_brief)
        elif all(isinstance(x, AudioFrame) for x in data):
            return _list_brief(data, _audio_frame_brief)
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return _list_brief(data, _ndarray_brief)
    return _default_brief(data)


def _ndarray_brief(array, maxlen=3):  # pragma: no cover
    head_vals = [repr(v) for i, v in enumerate(array.flatten()) if i < maxlen]
    if len(array.flatten()) > maxlen:
        head_vals.append('...')
    shape = 'shape=' + repr(array.shape)

    return '[' + ', '.join(head_vals) + '] ' + shape


def _image_brief(img):  # pragma: no cover
    return str(img)


def _audio_frame_brief(frame):  # pragma: no cover
    return str(frame)


def _text_brief(text, maxlen=128):  # pragma: no cover
    if len(text) > maxlen:
        return text[:maxlen] + '...'
    else:
        return text


def _list_brief(data, str_method, maxlen=4):  # pragma: no cover
    head_vals = [str_method(x) for i, x in enumerate(data) if i < maxlen]
    if len(data) > maxlen:
        head_vals.append('...')
        return '[' + ','.join(head_vals) + ']' + ' len=' + str(len(data))
    else:
        return '[' + ','.join(head_vals) + ']'


def _default_brief(data, maxlen=128):  # pragma: no cover
    s = str(data)
    return s[:maxlen] + '...' if len(s) > maxlen else s
