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

from towhee._types import Image


class DisplayMixin:
    """
    Mixin for display data.
    """

    def as_str(self):
        return self.factory(map(str, self._iterable))

    def show(self, limit=5, header=None, tablefmt='html'):
        """
        Print the first n lines of a DataCollection.

        Args:
            limit (`int`):
                The number of lines to print. Default value is 5.
                Print all if limit is non-positive.
            header (list of `str`):
                The field names.
            tablefmt (`str`):
                The format of the output, support html, plain.
        """
        # pylint: disable=import-outside-toplevel
        from towhee import Entity

        contents = [x for i, x in enumerate(self._iterable) if i < limit]

        if all(isinstance(x, Entity) for x in contents):
            header = tuple(contents[0].to_dict()) if not header else header
            data = [list(x.to_dict().values()) for x in contents]
        else:
            data = [[x] for x in contents]

        table_display(to_printable_table(data, header, tablefmt))


def table_display(table, tablefmt='html'):
    """
    Display table

    Args:
        table (`str`):
            Table in printable format, such as HTML.
        tablefmt (`str`):
            The format of the output, support html, plain.
    """
    # pylint: disable=import-outside-toplevel
    from IPython.display import display, HTML
    if tablefmt == 'html':
        display(HTML(table))
    elif tablefmt == 'plain':
        display(table)
    else:
        raise ValueError('unsupported table format %s' % tablefmt)


def to_printable_table(data, header=None, tablefmt='html'):
    """
    Convert two dimensional data structure into printable table

    Args:
        data (`List[List]`, or `List[Dict]`):
            The data filled into table. If a list of dict is given, keys are used as column names.
        header (`Iterable[str]`):
            The name of columns defined by users.
        tablefmt (`str`):
            The format of the output, support html, plain.
    """
    header = [] if not header else header

    if tablefmt == 'html':
        return to_html_table(data, header)
    elif tablefmt == 'plain':
        return to_plain_table(data, header, tablefmt)

    raise ValueError('unsupported table format %s' % tablefmt)


def to_plain_table(data, header, tablefmt):
    """
    Convert two dimensional data structure into plain table

    Args:
        data (`List[List]`, or `List[Dict]`):
            The data filled into table. If a list of dict is given, keys are used as column names.
        header (`Iterable[str]`):
            The name of columns defined by users.
        tablefmt (`str`):
            The format of the output, support plain.
    """
    # pylint: disable=import-outside-toplevel
    from tabulate import tabulate

    tb_contents = [[_to_plain_cell(x) for x in r] for r in data]
    return tabulate(tb_contents, headers=header, tablefmt=tablefmt)


def _to_plain_cell(data):
    # pylint: disable=import-outside-toplevel
    import numpy
    if isinstance(data, str):
        return _text_brief_repr(data)
    if isinstance(data, Image):
        return _image_brief_repr(data)
    elif isinstance(data, numpy.ndarray):
        return _ndarray_brief_repr(data)
    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return _texts_brief_repr(data)
        elif all(isinstance(x, Image) for x in data):
            return _images_brief_repr(data)
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return _ndarrays_brief_repr_(data)
    return _brief_repr(data)


def to_html_table(data, header):
    """
    Convert two dimensional data structure into html table

    Args:
        data (`List[List]`, or `List[Dict]`):
            The data filled into table. If a list of dict is given, keys are used as column names.
        header (`Iterable[str]`):
            The name of columns defined by users.
    """
    tb_style = 'style="border-collapse: collapse;"'
    th_style = 'style="text-align: center; font-size: 130%; border: none;"'
    td_style = 'style="text-align: center; border-right: solid 1px #D3D3D3; border-left: solid 1px #D3D3D3;"'

    trs = []
    trs.append('<tr>' + ' '.join(['<th ' + th_style + '>' + x + '</th>' for x in header]) + '</tr>')
    for r in data:
        trs.append('<tr>' + ' '.join(['<td ' + td_style + '>' + _to_html_cell(x) + '</td>' for x in r]) + '</tr>')
    return '<table ' + tb_style + '>' + ' '.join(trs) + '</table>'


def _to_html_cell(data):
    # pylint: disable=import-outside-toplevel
    import numpy
    if isinstance(data, str):
        return _text_brief_repr(data)
    if isinstance(data, Image):
        return _image_to_html_cell(data)
    elif isinstance(data, numpy.ndarray):
        return _ndarray_brief_repr(data)
    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return _texts_brief_repr(data)
        elif all(isinstance(x, Image) for x in data):
            return _images_to_html_cell(data)
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return _ndarrays_brief_repr_(data)
    return _brief_repr(data)


def _ndarray_brief_repr(array, maxlen=3):
    head_vals = [repr(v) for i, v in enumerate(array.flatten()) if i < maxlen]
    if len(array.flatten()) >= maxlen:
        head_vals.append('...')
    shape = 'shape=' + repr(array.shape)

    return '[' + ', '.join(head_vals) + '] ' + shape


def _ndarrays_brief_repr_(arrays, maxlen=3):
    return '[' + ', '.join([_ndarray_brief_repr(x, maxlen) for x in arrays]) + ']'


def _image_to_html_cell(img, width=128, height=128):
    # pylint: disable=import-outside-toplevel
    from towhee.utils.cv2_utils import cv2
    import base64

    _, img_encode = cv2.imencode('.JPEG', img)
    src = 'src="data:image/jpeg;base64,' + base64.b64encode(img_encode).decode() + '" '
    w = 'width = "' + str(width) + '" '
    h = 'height = "' + str(height) + '" '

    return '<img ' + src + w + h + '>'


def _images_to_html_cell(imgs, width=128, height=128):
    return ' '.join([_image_to_html_cell(x, width, height) for x in imgs])


def _images_brief_repr(imgs):
    return '[' + ', '.join([_image_brief_repr(x) for x in imgs]) + ']'


def _image_brief_repr(img):
    return _ndarray_brief_repr(img) + ' mode=' + img.mode


def _text_brief_repr(text, maxlen=32):
    if len(text) > maxlen:
        return text[:maxlen] + '...'
    else:
        return text


def _texts_brief_repr(texts, maxlen=32):
    return '[' + ', '.join([_text_brief_repr(x, maxlen) for x in texts]) + ']'


def _brief_repr(data, maxlen=128):
    s = repr(data)
    return s[:maxlen] + '...' if len(s) > maxlen else s
