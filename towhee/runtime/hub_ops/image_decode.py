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


class ImageDecode:
    """
    Image decode operators convert an encoded image back to its uncompressed format.
    In most cases, image decoding is the first step of an image processing pipeline.
    """

    cv2: HubOp = HubOp('image_decode.cv2')
    """
    `cv2 <https://towhee.io/image-decode/cv2>`_ is an image decode operator implementation with OpenCV.
    Can decode images from local files/HTTP urls and image binaries.

    Args:
       mode(`str`):
        BGR or RGB, default is BGR

    Examples:
      .. code-block:: python
        from towhee import pipe, ops, DataCollection

        # decode image, in bgr channel order
        p = (
          pipe.input('url')
          .map('url', 'image', ops.image_decode.cv2())
          .output('image')
        )

        # decode image, in rgb channel order
        p2 = (
          pipe.input('url')
          .map('url', 'image', ops.image_decode.cv2('rgb'))
          .output('image')
        )

        # decode from path
        DataCollection(p('./src_dog.jpg')).show()

        # decode from bytes
        with open('./src_dog.jpg', 'rb') as f:
           DataCollection(p2(f.read())).show()
    """

    nvjpeg: HubOp = HubOp('image_decode.nvjpeg')
    """
    `nvjpeg <https://towhee.io/image-decode/nvjpeg>`_ is an image decode operator
    implementation with OpenCV and nvjpeg.
    In CPU env, use OpenCV, in GPU env, use nvjpeg to decode jpeg files.

    Can decode images from local files/HTTP urls and image binaries.

    Args:
        device(`int`):
            GPU ID, default is 0.

    Examples:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('url')
            .map('url', 'image', ops.image_decode.nvjpeg())
            .output('image')
        )

        DataCollection(p('./src_dog.jpg')).show()
    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.image_decode')(*args, **kwargs)
