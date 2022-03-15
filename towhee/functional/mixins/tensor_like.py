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


class TensorLikeMixin:
    """Mixin for tensor like data.
    """

    # pylint: disable=import-outside-toplevel
    def stack(self, axis=0):
        """stack the sequence of inputs along a new axis.

        Args:
            axis (int, optional): the axis of the result array along which the inputs are
                stacked. Defaults to 0.

        Returns:
            ndarray: the stacked array.

        Examples:

        >>> import numpy as np
        >>> from towhee.functional import DataCollection
        >>> dc = (
        ...     DataCollection.range(10)
        ...         .map(lambda x: np.array([x]))
        ...         .batch(2)
        ...         .stack(axis=1)
        ... )
        >>> dc.to_list()
        [array([[0, 1]]), array([[2, 3]]), array([[4, 5]]), array([[6, 7]]), array([[8, 9]])]
        """
        import numpy as np

        def inner(x):
            return np.stack(x, axis=axis)

        return self.factory(map(inner, self._iterable))

    def unstack(self, axis=0):
        """unstack an array along given axis.

        Args:
            axis (int, optional): the axis along which to unstack the array. Defaults to 0.

        Returns:
            [ndarray]: sequence of result arrays.

        Examples:

        >>> import numpy as np
        >>> from towhee.functional import DataCollection
        >>> dc = (
        ...     DataCollection.range(3)
        ...         .map(lambda x: np.array([x, x*2]))
        ...         .unstack(axis=0)
        ... )
        >>> dc.to_list()
        [[array(0), array(0)], [array(1), array(2)], [array(2), array(4)]]
        """
        import numpy as np

        def inner(x):
            split = x.shape[axis]
            shape = x.shape
            shape = [shape[i] for i in range(len(shape)) if i != axis]
            # if shape:
            return list(map(lambda arr: arr.reshape(shape), np.split(x, split, axis=axis)))
            # else:
                # return np.split(x, split, axis=axis)

        return self.factory(map(inner, self._iterable))

    def concat(self, axis=0):
        """concat the sequence of inputs along a axis (no new axis)

        Args:
            axis (int, optional): the axis alone which to concat inputs. Defaults to 0.

        Returns:
            ndarray: the stacked array.

        Examples:

        >>> import numpy as np
        >>> from towhee.functional import DataCollection
        >>> dc = (
        ...     DataCollection.range(10)
        ...         .map(lambda x: np.array([x]))
        ...         .batch(2)
        ...         .concat(axis=0)
        ... )
        >>> dc.to_list()
        [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7]), array([8, 9])]
        """
        import numpy as np

        def inner(x):
            return np.concatenate(list(x), axis=axis)

        return self.factory(map(inner, self._iterable))

    def split(self, axis=0):
        """split the input array along a axis.

        Args:
            axis (int, optional): the axis along which to split the array. Defaults to 0.

        Returns:
            [ndarray]: list of result arrays.

        Examples:

        >>> import numpy as np
        >>> from towhee.functional import DataCollection
        >>> dc = (
        ...     DataCollection.range(3)
        ...         .map(lambda x: np.array([[x, x*2]]))
        ...         .split(axis=1)
        ... )
        >>> dc.to_list()
        [[array([[0]]), array([[0]])], [array([[1]]), array([[2]])], [array([[2]]), array([[4]])]]
        """
        import numpy as np

        def inner(x):
            split = x.shape[axis]
            return np.split(x, split, axis=axis)

        return self.factory(map(inner, self._iterable))

    def normalize(self, axis=0):
        import numpy as np

        def inner(x):
            return np.linalg.norm(x, axis=axis)

        return self.factory(map(inner, self._iterable))

    def reshape(self, shape):

        def inner(x):
            return x.reshape(shape)

        return self.factory(map(inner, self._iterable))


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest
    doctest.testmod(verbose=False)
