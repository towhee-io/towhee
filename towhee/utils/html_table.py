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
from towhee.utils.thirdparty.ipython_utils import display, HTML

from towhee.types import Image, VideoFrame, AudioFrame


class NestedHTMLTable:
    """Convert nested  data structure into html table

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
    def to_html_table(data, headers):
        """Convert two dimensional data structure into html table

        Args:
            data (List)
            headers (Iterable[str]): The names of columns defined by user.
        Returns:
            str: The table in html format.
        """
        tb_style = 'style="border-collapse: collapse;"'
        th_style = 'style="text-align: center; font-size: 130%; border: none;"'
        trs = []
        trs.append('<tr>' + ' '.join(['<th ' + th_style + '>' + x + '</th>' for x in headers]) + '</tr>')

        for r in data:
            tds = []
            for x in r:
                if isinstance(x, dict) and 'headers' in x and 'data' in x:
                    tds.append('<td>' + NestedHTMLTable.to_html_table(x['data'], x['headers']) + '</td>')
                else:
                    tds.append(_to_html_td(x))

            trs.append(
                '<tr>' + ''.join(tds) + '</tr>'
            )
        return '<table ' + tb_style + '>' + '\n'.join(trs) + '</table>'

    def show(self):
        table = NestedHTMLTable.to_html_table(self._data['data'], self._data['headers'])
        display(HTML(table))


def _to_html_td(data):  # pragma: no cover

    def wrap_td_tag(content, align='center', vertical_align='top'):
        td_style = 'style="' \
            + 'text-align: ' + align + '; ' \
            + 'vertical-align: ' + vertical_align + '; ' \
            + 'border-right: solid 1px #D3D3D3; ' \
            + 'border-left: solid 1px #D3D3D3; ' \
            + '"'
        return '<td ' + td_style + '>' + content + '</td>'

    if isinstance(data, str):
        return wrap_td_tag(_text_brief(data))
    if isinstance(data, (Image, VideoFrame)):
        return wrap_td_tag(_image_to_html_cell(data))
    elif isinstance(data, AudioFrame):
        return wrap_td_tag(_audio_frame_to_html_cell(data))
    elif isinstance(data, numpy.ndarray):
        return wrap_td_tag(_ndarray_brief(data), align='left')

    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, str) for x in data):
            return wrap_td_tag(_text_list_brief(data))
        elif all(isinstance(x, (Image, VideoFrame)) for x in data):
            return wrap_td_tag(_images_to_html_cell(data), vertical_align='top')
        elif all(isinstance(x, AudioFrame) for x in data):
            return wrap_td_tag(_audio_frames_to_html_cell(data), vertical_align='top')
        elif all(isinstance(x, numpy.ndarray) for x in data):
            return wrap_td_tag(_list_brief(data, _ndarray_brief), align='left')
        else:
            return wrap_td_tag(_list_brief(data, _default_brief), align='left')
    return wrap_td_tag(_default_brief(data))


def _image_to_html_cell(img, width=128, height=128):  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    from towhee.utils.cv2_utils import cv2
    from towhee.utils.matplotlib_utils import plt
    import base64
    from io import BytesIO
    plt.ioff()
    fig = plt.figure(figsize=(width / 100, height / 100))
    img = cv2.resize(img, (width, height))
    fig.figimage(img, resize=True)
    plt.ion()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    data = base64.b64encode(tmpfile.getvalue()).decode('ascii')
    src = 'src="data:image/png;base64,' + data + '" '
    w = 'width = "' + str(128) + 'px" '
    h = 'height = "' + str(128) + 'px" '
    style = 'style = "float:left; padding:2px"'
    return '<img ' + src + w + h + style + '>'


def _images_to_html_cell(imgs, width=128, height=128):  # pragma: no cover
    return ' '.join([_image_to_html_cell(x, width, height) for x in imgs])


def _audio_frame_to_html_cell(frame, width=128, height=128):  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    from towhee.utils.matplotlib_utils import plt

    signal = frame[0, ...]
    fourier = numpy.fft.fft(signal)
    freq = numpy.fft.fftfreq(signal.shape[-1], 1)
    fig = plt.figure()
    plt.plot(freq, fourier.real)
    fig.canvas.draw()
    data = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image(data, 'RGB')
    plt.close()
    return _image_to_html_cell(img, width, height)


def _audio_frames_to_html_cell(frames, width=128, height=128):  # pragma: no cover
    return ' '.join([_audio_frame_to_html_cell(x, width, height) for x in frames])


def _video_path_to_html_cell(path, width=128, height=128):
    src = 'src="' + path + '" '
    type_suffix_idx = path.rfind('.')
    if type_suffix_idx == -1:
        raise ValueError('unsupported video format %s' % path)

    video_type = 'type="video/' + path[path.rfind('.') + 1:].lower() + '" '
    w = 'width = "' + str(width) + 'px" '
    h = 'height = "' + str(height) + 'px" '
    style = 'style = "float:left; padding:2px"'

    return '<video ' + w + h + style + 'controls><source ' + src + video_type + '></source></video>'


def _ndarray_brief(array, maxlen=3):  # pragma: no cover
    head_vals = [repr(v) for i, v in enumerate(array.flatten()) if i < maxlen]
    if len(array.flatten()) > maxlen:
        head_vals.append('...')
    shape = 'shape=' + repr(array.shape)

    return '[' + ', '.join(head_vals) + '] ' + shape


def _text_brief(text, maxlen=128):  # pragma: no cover
    if len(text) > maxlen:
        return text[:maxlen] + '...'
    else:
        return text


def _text_list_brief(data, maxlen=16):  # pragma: no cover
    head_vals = ['<br>' + _text_brief(x) + '</br>' for i, x in enumerate(data) if i < maxlen]
    if len(data) > maxlen:
        head_vals.append('<br>...</br>')
    return ' '.join(head_vals)


def _list_brief(data, str_method, maxlen=4):  # pragma: no cover
    head_vals = [str_method(x) for i, x in enumerate(data) if i < maxlen]
    if len(data) > maxlen:
        head_vals.append('...')
    return '[' + ','.join(head_vals) + ']' + ' len=' + str(len(data))


def _default_brief(data, maxlen=128):  # pragma: no cover
    s = str(data)
    return s[:maxlen] + '...' if len(s) > maxlen else s
