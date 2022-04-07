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

from towhee.engine import register

# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


@register(name='builtin/tensor_hstack')
def tensor_hstack(*arg):
    from scipy import sparse
    return sparse.hstack(arg)


@register(name='builtin/stack')
class stack:
    """
    Stack the sequence of inputs along a new axis.

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

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np

        return np.stack(x, axis=self._axis)


@register(name='builtin/unstack')
class unstack:
    """
    Unstack an array along given axis.

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

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np

        nsplit = x.shape[self._axis]
        shape = x.shape
        shape = [shape[i] for i in range(len(shape)) if i != self._axis]
        return list(
            map(lambda arr: arr.reshape(shape),
                np.split(x, nsplit, axis=self._axis)))


@register(name='builtin/concat')
class concat:
    """
    Concat the sequence of inputs along a axis (no new axis)

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

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np
        return np.concatenate(list(x), axis=self._axis)


@register(name='builtin/split')
class split:
    """
    Split the input array along a axis.

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

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np

        nsplit = x.shape[self._axis]
        return np.split(x, nsplit, axis=self._axis)


@register(name='builtin/normalize')
class normalize:
    """
    Normalize input tensor.

    Examples:

    >>> import numpy
    >>> from towhee.functional import DataCollection
    >>> dc = DataCollection([numpy.array([3, 4]), numpy.array([6,8])])
    >>> dc.normalize().to_list()
    [array([0.6, 0.8]), array([0.6, 0.8])]
    """

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np

        return x / np.linalg.norm(x, axis=self._axis)


@register(name='builtin/reshape')
class reshape:

    def __init__(self, shape):
        self._shape = shape

    def __call__(self, x):

        return x.reshape(self._shape)


@register(name='builtin/random')
class random:
    """
    Return a random tensor filled with random numbers from [0.0, 1.0).

    Args:
        shape ([int], optional): tensor shape. Defaults to None.

    Returns:
        ndarray: output tensor.

    Examples:

    >>> from towhee.functional import DataCollection
    >>> DataCollection.range(5).random(shape=[1,2,3]).map(lambda x: x.shape).to_list()
    [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)]
    """

    def __init__(self, shape):
        self._shape = shape

    def __call__(self, _):
        import numpy as np

        return np.random.random(self._shape)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
