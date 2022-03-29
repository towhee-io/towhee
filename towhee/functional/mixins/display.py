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
from typing import Iterable

from tabulate import tabulate
from towhee._types import Image


class DisplayMixin:
    """
    Mixin for display data.
    """

    def as_str(self):
        return self.factory(map(str, self._iterable))

    def plot(self):
        pass

    def head(self, num: int = 5, header: Iterable[str] = None, numalign: str = 'center', stralign: str = 'center', tablefmt: str = 'plain'):
        """
        Print the first n lines in a DataCollection.

        Args:
            num (`int`):
                The number of lines to print. Default value is 5.
            header (`Iterable[str]`):
                The name of columns defined by users.
            numalign (`str`):
                How the nums align, support center, right, left.
            stralign (`str`):
                How the strs align, support center, right, left.
            tablefmt (`str`):
                The format of the output, support plain, simple, github, grid, fancy_grid, pipe, orgtbl, jira, presto, psql, rst, mediawiki,
                moinmoin, youtrack, html, latex, latex_raw, latex_booktabs, textile.
        """
        to_display = []

        cnt = 0
        for i in self._iterable:
            header = i.info[1:] if not header else header
            to_display.append(i.to_dict().values())
            cnt += 1
            if cnt == num:
                break

        print(tabulate(to_display, headers=header, tablefmt=tablefmt, numalign=numalign, stralign=stralign))

    def tail(self, num: int = 5, header: Iterable[str] = None, numalign: str = 'center', stralign: str = 'center', tablefmt: str = 'plain'):
        """
        Print the last n lines in a DataCollection.

        Args:
            num (`int`):
                The number of lines to print. Default value is 5.
            header (`Iterable[str]`):
                The name of columns defined by users.
            numalign (`str`):
                How the nums align, support center, right, left.
            stralign (`str`):
                How the strs align, support center, right, left.
            tablefmt (`str`):
                The format of the output, support plain, simple, github, grid, fancy_grid, pipe, orgtbl, jira, presto, psql, rst, mediawiki,
                moinmoin, youtrack, html, latex, latex_raw, latex_booktabs, textile.
        """
        to_display = []

        if self.is_stream:
            raise AttributeError('The DataCollection is stream, tail function not supported.')

        maxsize = len(self._iterable)
        for i in self._iterable[maxsize - num:]:
            header = i.info[1:] if not header else header
            to_display.append(i.to_dict().values())

        print(tabulate(to_display, headers=header, tablefmt=tablefmt, numalign=numalign, stralign=stralign))


def to_printable_table(data, header, tablefmt='html'):
    """
    Convert two dimensional data structure into printable table

    Args:
        data (`List[List]`, or `List[Dict]`):
            The data filled into table. If a list of dict is given, keys are used as column names.
        header (`Iterable[str]`):
            The name of columns defined by users.
        tablefmt (`str`):
            The format of the output, support html.
    """
    if tablefmt == 'html':
        return to_html_table(data, header)

    raise ValueError('unsupported table format %s' % tablefmt)


def to_html_table(data, header):
    """
    Convert two dimensional data structure into html table

    Args:
        data (`List[List]`, or `List[Dict]`):
            The data filled into table. If a list of dict is given, keys are used as column names.
        header (`Iterable[str]`):
            The name of columns defined by users.
    """

    tb_contents = []
    for r in data:
        tb_contents.append([_to_html_cell(x) for x in r])
    return tabulate(tb_contents, headers=header, tablefmt='html')


def _to_html_cell(data):
    # pylint: disable=import-outside-toplevel
    import numpy
    if isinstance(data, str):
        return _text_to_html_cell(data)
    if isinstance(data, Image):
        return _image_to_html_cell(data)
    elif isinstance(data, numpy.ndarray):
        return _ndarray_to_html_cell(data)
    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return _texts_to_html_cell(data)
        elif all(isinstance(x, Image) for x in data):
            return _images_to_html_cell(data)
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return _ndarrays_to_html_cell(data)
    return repr(data)


def _ndarray_to_html_cell(array, maxlen=4):
    head_vals = [repr(v) for i, v in enumerate(array.flatten()) if i < maxlen]
    if len(array.flatten()) >= maxlen:
        head_vals.append('...')
    shape = 'shape=' + repr(array.shape)

    return '[' + ', '.join(head_vals) + '] ' + shape


def _ndarrays_to_html_cell(arrays, maxlen=4):
    return '[' + ', '.join([_ndarray_to_html_cell(x, maxlen) for x in arrays]) + ']'


def _image_to_html_cell(img, width=200, height=200):
    # pylint: disable=import-outside-toplevel
    import cv2
    import base64

    img_encode = cv2.imencode('.jpeg', img)[1]
    src = 'src="data:image/jpeg;base64,' + base64.b64encode(img_encode) + '" '
    w = 'width = "' + str(width) + '" '
    h = 'height = "' + str(height) + '" '

    return '<img ' + src + w + h + '>'


def _images_to_html_cell(imgs, width=200, height=200):
    return ' '.join([_image_to_html_cell(x, width, height) for x in imgs])


def _text_to_html_cell(text, maxlen=32):
    if len(text) > maxlen:
        return text[:maxlen] + '...'
    else:
        return text


def _texts_to_html_cell(texts, maxlen=32):
    return '[' + ', '.join([_text_to_html_cell(x, maxlen) for x in texts]) + ']'
