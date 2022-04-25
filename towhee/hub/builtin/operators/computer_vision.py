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
import uuid
from pathlib import Path
import numpy as np

from towhee.engine import register
from towhee._types.image import Image
from towhee.types.image_utils import to_image_color

# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name
# pylint: disable=redefined-builtin


@register(name='builtin/image_load')
def image_load(file_or_buff):
    """load image from paths, file objects or memory blocks.

    Returns:
        ndarray: output image

    Examples:

    >>> from towhee.hub import preclude
    >>> from towhee.functional import DataCollection
    >>> dc = (
    ...     DataCollection.range(5)
    ...         .tensor_random(shape=[100, 100, 3])
    ...         .image_dump()
    ... ).to_list()

    >>> (
    ...     DataCollection(dc).image_load()
    ...         .map(lambda x: x.shape)
    ... ).to_list()
    [(100, 100, 3), (100, 100, 3), (100, 100, 3), (100, 100, 3), (100, 100, 3)]
    """
    from towhee.utils.cv2_utils import cv2

    if isinstance(file_or_buff, str):
        return cv2.imread(file_or_buff)
    if hasattr(file_or_buff, 'read'):
        buff = np.frombuffer(file_or_buff.read(), dtype=np.uint8)
    else:
        buff = np.frombuffer(file_or_buff, dtype=np.uint8)
    return cv2.imdecode(buff, 1)


@register(name='builtin/image_dump')
class image_dump:
    """dump image to a binary buffer.
    """

    def __init__(self, ext='.JPEG'):
        self._ext = ext

    def __call__(self, x):
        from towhee.utils.cv2_utils import cv2

        retval, im = cv2.imencode(self._ext, x)
        if retval >= 0:
            return im
        else:
            raise Exception('error while encoding image: {}'.format(retval))


@register(name='builtin/image_resize')
class image_resize:
    """resize an image.

    Args:
        dsize ((int, int), optional): target image size. Defaults to None.
        fx (float, optional): scale factor for x axis. Defaults to None.
        fy (float, optional): scale factor for y axis. Defaults to None.
        interpolation (str|int, optional): interpolation method, see detailed document for `cv2.resize`. Defaults to None.

    Returns:
        ndarray: output image.

    Examples:

    >>> from towhee.functional import DataCollection
    >>> dc = (
    ...     DataCollection.range(5)
    ...         .tensor_random(shape=[100, 100, 3])
    ...         .image_resize(dsize=[10, 10], interpolation='nearest')
    ... )
    >>> dc.select['shape']().as_raw().to_list()
    [(10, 10, 3), (10, 10, 3), (10, 10, 3), (10, 10, 3), (10, 10, 3)]
    """

    def __init__(self, dsize=None, fx=None, fy=None, interpolation=None):
        from towhee.utils.cv2_utils import cv2

        self._dsize = dsize
        self._fx = fx
        self._fy = fy
        if interpolation is not None and isinstance(interpolation, str):
            interpolation = 'INTER_{}'.format(interpolation).upper()
            if hasattr(cv2, interpolation):
                interpolation = getattr(cv2, interpolation)
            else:
                interpolation = None
        self._interpolation = interpolation

    def __call__(self, x):
        from towhee.utils.cv2_utils import cv2
        return cv2.resize(x,
                          dsize=self._dsize,
                          fx=self._fx,
                          fy=self._fy,
                          interpolation=self._interpolation)


@register(name='builtin/image_convert_color')
class image_convert_color:
    """convert image color space.

    Args:
        code (str|int): color space conversion string or code

    Returns:
        ndarray: output image.

    Examples:

    >>> from towhee.functional import DataCollection
    >>> import numpy as np
    >>> (
    ...     DataCollection([np.ones([1,1, 3], dtype=np.uint8)])
    ...         .image_convert_color(code='rgb2gray')
    ...         .to_list()
    ... )
    [array([[1]], dtype=uint8)]
    """

    def __init__(self, code):
        from towhee.utils.cv2_utils import cv2

        if code is not None and isinstance(code, str):
            code = 'COLOR_{}'.format(code).upper()
            if hasattr(cv2, code):
                code = getattr(cv2, code)
            else:
                code = None
        self._code = code

    def __call__(self, x):
        from towhee.utils.cv2_utils import cv2
        return cv2.cvtColor(x, self._code)


@register(name='builtin/image_filter')
class image_filter:
    """image filter.
    """

    def __init__(self, ddepth, kernel):
        self._ddepth = ddepth
        self._kernel = kernel

    def __call__(self, x):

        from towhee.utils.cv2_utils import cv2

        return cv2.filter2D(x, self._ddepth, self._kernel)


@register(name='builtin/image_blur')
class image_blur:
    """image blur.

    Args:
        ksize ([int]): kernel size, for example: [3, 3].

    Returns:
        ndarray: output image

    >>> from towhee.functional import DataCollection
    >>> import numpy as np
    >>> (
    ...     DataCollection([np.ones([5,5], dtype=np.uint8)])
    ...         .image_blur(ksize=[3,3])
    ...         .to_list()
    ... )
    [array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]], dtype=uint8)]
    """

    def __init__(self, ksize):
        self._ksize = ksize

    def __call__(self, x):
        from towhee.utils.cv2_utils import cv2

        return cv2.blur(x, self._ksize)


@register(name='builtin/save_image')
class save_image:
    """
    Save image to specific directory with the uuid name.

    Args:
        dir (`str`):
            The directory to save image.

    Examples:

    >>> import towhee
    >>> (
    ...     towhee.dc['path'](['https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'])
    ...           .image_decode['path', 'img']()
    ...           .save_image['img','path_new'](dir='temp/pic')
    ...           .to_list()
    ... )
    [<Entity dict_keys(['path', 'img', 'path_new'])>]
    """

    def __init__(self, dir: str):
        self._file = str(uuid.uuid4()) + '.jpg'
        self._img_path = str(Path(dir) / self._file)
        if not Path(dir).exists():
            Path(dir).mkdir(parents=True)

    def __call__(self, img):
        from towhee.utils.pil_utils import PILImage
        from towhee.utils.ndarray_utils import cv2
        if isinstance(img, PILImage.Image):
            img = to_image_color(img, 'RGB')
            img.save(self._img_path)
        elif isinstance(img, (Image, np.ndarray)):
            img = to_image_color(img, 'BGR')
            cv2.imwrite(self._img_path, img)
        return self._img_path


__test__ = {'image_load': image_load.__doc__}

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
