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
from typing import Tuple

from towhee._types import Image
from towhee.types import AudioFrame
from towhee.hparam import param_scope
from towhee.functional.entity import Entity

# pylint: disable=dangerous-default-value


def get_df_on_columns(self, index: Tuple[str]):
    # pylint: disable=import-outside-toplevel
    from towhee.utils.pandas_utils import pandas as pd

    def inner(entity: Entity):
        data = {}
        for feature in index:
            data[feature] = getattr(entity, feature)
        return data

    data = map(inner, self)
    df = pd.DataFrame(data)
    return df


def calc_df(df, feature: str, target: str):
    # pylint: disable=import-outside-toplevel
    from towhee.utils.pandas_utils import pandas as pd
    lst = []
    df[feature] = df[feature].fillna('NULL')

    if target:
        for i in range(df[feature].nunique()):
            val = list(df[feature].unique())[i]
            lst.append([feature,                                                         # Variable
                        val,                                                             # Value
                        df[df[feature] == val].count()[feature],                         # All
                        df[(df[feature] == val) & (df[target] == 0)].count()[feature],   # Good (think: Fraud == 0)
                        df[(df[feature] == val) & (df[target] == 1)].count()[feature]])  # Bad (think: Fraud == 1)

        data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
        data['Share'] = data['All'] / data['All'].sum()
        data['Bad Rate'] = data['Bad'] / data['All']
        data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
        data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
        data['WoE'] = numpy.log(data['Distribution Good'] / data['Distribution Bad'])

        data = data.replace({'WoE': {numpy.inf: 0, -numpy.inf: 0}})
        data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

        data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
        data.index = range(len(data.index))

        iv = data['IV'].sum()
        print(f'Variable: {feature}\'s IV sum is: {iv}')
        return data
    else:
        for i in range(df[feature].nunique()):
            val = list(df[feature].unique())[i]
            lst.append([feature,                                                         # Variable
                        val,                                                             # Value
                        df[df[feature] == val].count()[feature]])                        # All
        data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All'])
        data['Share'] = data['All'] / data['All'].sum()
        return data


def _feature_summarize_callback(self):
    # pylint: disable=import-outside-toplevel
    from towhee.utils.pandas_utils import pandas as pd

    def wrapper(_: str, index, *arg, **kws):
        if isinstance(index, str):
            index = (index,)
        index = list(index)
        if arg:
            kws['target'], = arg

        target = None
        if 'target' in kws:
            target = kws.pop('target')
            index.append(target)

        df = get_df_on_columns(self, index)
        summarize = pd.DataFrame()
        if target:
            index.remove(target)
        for feature in index:
            data = calc_df(df, feature, target)
            summarize = summarize.append(data)

        # pylint: disable=import-outside-toplevel
        from towhee.utils import ipython_utils
        ipython_utils.display(summarize)

    return wrapper


def _plot_callback(self):
    # pylint: disable=unused-argument
    def wrapper(_: str, index, *arg, **kws):
        if isinstance(index, str):
            index = (index,)
        df = get_df_on_columns(self, index)
        if 'kind' not in kws:
            kws.update(kind='hist')
        df.plot(**kws)

    return wrapper


class DisplayMixin:
    """
    Mixin for display data.
    """

    def __init__(self):
        super().__init__()
        self.feature_summarize = param_scope().callholder(_feature_summarize_callback(self))
        self.plot = param_scope().callholder(_plot_callback(self))

    def as_str(self):
        return self._factory(map(str, self._iterable))

    def show(self, limit=5, header=None, tablefmt='html', formatter={}):
        """
        Print the first n lines of a DataCollection.

        Args:
            limit (`int`):
                The number of lines to print. Default value is 5.
                Print all if limit is non-positive.
            header (list of `str`):
                The field names.
            tablefmt (`str`):
                The format of the output, support html, plain, grid.
        """

        contents = [x for i, x in enumerate(self._iterable) if i < limit]

        if all(isinstance(x, Entity) for x in contents):
            header = tuple(contents[0].to_dict()) if not header else header
            data = [list(x.to_dict().values()) for x in contents]
        else:
            data = [[x] for x in contents]

        table_display(to_printable_table(data, header, tablefmt, formatter))


def table_display(table, tablefmt='html'):
    """
    Display table

    Args:
        table (`str`):
            Table in printable format, such as HTML.
        tablefmt (`str`):
            The format of the output, support html, plain, grid.
    """
    # pylint: disable=import-outside-toplevel
    from towhee.utils import ipython_utils

    if tablefmt == 'html':
        ipython_utils.display(ipython_utils.HTML(table))
    elif tablefmt in ('plain', 'grid'):
        print(table)
    else:
        raise ValueError('unsupported table format %s' % tablefmt)


def to_printable_table(data, header=None, tablefmt='html', formatter={}):
    """
    Convert two dimensional data structure into printable table

    Args:
        data (`List[List]`, or `List[Dict]`):
            The data filled into table. If a list of dict is given, keys are used as column names.
        header (`Iterable[str]`):
            The name of columns defined by users.
        tablefmt (`str`):
            The format of the output, support html, plain, grid.
    """
    header = [] if not header else header

    if tablefmt == 'html':
        return to_html_table(data, header, formatter)
    elif tablefmt in ('plain', 'grid'):
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
            The format of the output, support plain, grid.
    """
    # pylint: disable=import-outside-toplevel
    from tabulate import tabulate

    tb_contents = [[_to_plain_cell(x) for x in r] for r in data]
    return tabulate(tb_contents, headers=header, tablefmt=tablefmt)


def _to_plain_cell(data):
    if isinstance(data, str):
        return _text_brief(data)
    if isinstance(data, Image):
        return _image_brief(data)
    elif isinstance(data, AudioFrame):
        return _audio_frame_brief(data)
    elif isinstance(data, numpy.ndarray):
        return _ndarray_brief(data)

    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return _list_brief(data, _text_brief)
        elif all(isinstance(x, Image) for x in data):
            return _list_brief(data, _image_brief)
        elif all(isinstance(x, AudioFrame) for x in data):
            return _list_brief(data, _audio_frame_brief)
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return _list_brief(data, _ndarray_brief)
    return _default_brief(data)


def to_html_table(data, header, formatter={}):
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

    str_2_callback = {
        'text': _text_brief,
        'image': _image_to_html_cell
    }

    trs = []
    trs.append('<tr>' + ' '.join(['<th ' + th_style + '>' + x + '</th>' for x in header]) + '</tr>')

    to_html_callback = {}
    for i, field in enumerate(header):
        cb = formatter.get(field, None)
        if cb is None:
            cb = _to_html_cell
        elif isinstance(cb, str):
            cb = str_2_callback.get(cb, _to_html_cell)
        to_html_callback[i] = cb
    for r in data:
        trs.append(
            '<tr>' + ' '.join(['<td ' + td_style + '>' + to_html_callback.get(i, _to_html_cell)(x) + '</td>' for i, x in enumerate(r)]) + '</tr>'
        )
    return '<table ' + tb_style + '>' + ' '.join(trs) + '</table>'


def _to_html_cell(data):
    if isinstance(data, str):
        return _text_brief(data)
    if isinstance(data, Image):
        return _image_to_html_cell(data)
    elif isinstance(data, AudioFrame):
        return _audio_frame_brief(data)
    elif isinstance(data, numpy.ndarray):
        return _ndarray_brief(data)

    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return _list_brief(data, _text_brief)
        elif all(isinstance(x, Image) for x in data):
            return _images_to_html_cell(data)
        elif all(isinstance(x, AudioFrame) for x in data):
            return _list_brief(data, _audio_frame_brief)
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return _list_brief(data, _ndarray_brief)
    return _default_brief(data)


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


def _ndarray_brief(array, maxlen=3):
    head_vals = [repr(v) for i, v in enumerate(array.flatten()) if i < maxlen]
    if len(array.flatten()) > maxlen:
        head_vals.append('...')
    shape = 'shape=' + repr(array.shape)

    return '[' + ', '.join(head_vals) + '] ' + shape


def _image_brief(img):
    return str(img)


def _audio_frame_brief(frame):
    return str(frame)


def _text_brief(text, maxlen=32):
    if len(text) > maxlen:
        return text[:maxlen] + '...'
    else:
        return text


def _list_brief(data, str_method, maxlen=4):
    head_vals = [str_method(x) for i, x in enumerate(data) if i < maxlen]
    if len(data) > maxlen:
        head_vals.append('...')
    return '[' + ','.join(head_vals) + ']' + ' len=' + str(len(data))


def _default_brief(data, maxlen=128):
    s = repr(data)
    return s[:maxlen] + '...' if len(s) > maxlen else s


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
