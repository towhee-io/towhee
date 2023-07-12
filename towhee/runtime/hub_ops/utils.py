# Copyright 2023 Zilliz. All rights reserved.
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

from towhee.runtime.factory import HubOp


class Utils:
    """
    Some utils.
    """

    np_normalize: HubOp = HubOp('towhee.np_normalize')
    """
    Convert the ndarray to a unit vector.

    __init__(self, axis=0)
        if the axis is an integer, then the vector norm is computed for the axis of x.
        If the axis is a 2-tuple, the matrix norms of specified matrices are computed.
        If the axis is None, then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.

    __call__(self, x: ndarray)
        x(`ndarray`)
            input ndarray

    Example:

    .. code-block:: python

        import numpy as np
        from towhee.dc2 import pipe, ops

        p = (
            pipe.input('vec')
            .map('vec', 'vec', ops.utils.np_normalize())
            .output('vec')
            )

        p(np.random.rand(20)).to_list()

    """

    image_crop: HubOp = HubOp('towhee.image_crop')
    """
    An image crop operator implementation with OpenCV.

    __init__(self, clamp = False)
        clamp(`bool`):
            If set True, coordinates of bounding boxes would be clamped into image size.


    __call__(self, img: np.ndarray, bboxes: List[Tuple]) -> List[ndarray]
        img(`ndarray`):
            The image need to be cropped.
        bboxes(`List[Tuple[int, int, int, int]]`):
            The nx4 numpy tensor for n bounding boxes need to crop, each row is formatted as (x1, y1, x2, y2).

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('path')
                .map('path', 'img', ops.image_decode.cv2('rgb'))
                .map('img', ('box','score'), ops.face_detection.retinaface())
                .map(('img', 'box'), 'crop', ops.utils.image_crop(clamp = True))
                .output('img', 'crop')
        )

        DataCollection(p('./avengers.jpg')).show()
    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.utils')(*args, **kwargs)
