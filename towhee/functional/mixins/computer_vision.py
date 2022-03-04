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


def _normalize(x):
    import numpy as np # pylint: disable=import-outside-toplevel
    return x / np.linalg.norm(x)


class ComputerVisionMixin:
    """
    Mixin for computer vision problems.

    Examples:
    # >>> from towhee.functional import DataCollection
    # >>> DataCollection.from_camera(1).imshow()
    """

    def normalize(self, name=None):
        """
        normalize the output embedding

        Examples:
        >>> from typing import NamedTuple
        >>> import numpy
        >>> from towhee.functional import DataCollection
        >>> Outputs = NamedTuple('Outputs', [('feature_vector', numpy.array)])
        >>> dc = DataCollection([Outputs(numpy.array([3, 4])), Outputs(numpy.array([6,8]))])
        >>> dc.normalize(name='feature_vector').to_list()
        [array([0.6, 0.8]), array([0.6, 0.8])]
        """
        if name is not None:
            retval = self.select(name).map(_normalize)
        else:
            retval = self.map(_normalize)
        return self.factory(retval)

    def imshow(self, title="image"):
        import cv2  # pylint: disable=import-outside-toplevel
        for im in self:
            cv2.imshow(title, im)
            cv2.waitKey(1)


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest
    doctest.testmod(verbose=False)
