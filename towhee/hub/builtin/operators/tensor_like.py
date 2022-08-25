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
    from towhee.utils.scipy_utils import sparse
    return sparse.hstack(arg)


@register(name='builtin/tensor_stack')
class tensor_stack:
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
    ...         .tensor_stack(axis=1)
    ... )
    >>> dc.to_list()
    [array([[0, 1]]), array([[2, 3]]), array([[4, 5]]), array([[6, 7]]), array([[8, 9]])]
    """

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np

        return np.stack(x, axis=self._axis)


@register(name='builtin/tensor_unstack')
class tensor_unstack:
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
    ...         .tensor_unstack(axis=0)
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


@register(name='builtin/tensor_concat')
class tensor_concat:
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
    ...         .tensor_concat(axis=0)
    ... )
    >>> dc.to_list()
    [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7]), array([8, 9])]
    """

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np
        return np.concatenate(list(x), axis=self._axis)


@register(name='builtin/tensor_split')
class tensor_split:
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
    ...         .tensor_split(axis=1)
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


@register(name='builtin/tensor_normalize')
class tensor_normalize:
    """
    Normalize input tensor.

    Examples:

    >>> import numpy
    >>> from towhee.functional import DataCollection
    >>> dc = DataCollection([numpy.array([3, 4]), numpy.array([6,8])])
    >>> dc.tensor_normalize().to_list()
    [array([0.6, 0.8]), array([0.6, 0.8])]
    """

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, x):
        import numpy as np

        return x / np.linalg.norm(x, axis=self._axis)


@register(name='builtin/tensor_reshape')
class tensor_reshape:
    """
    Reshape input tensor.

    Examples:

    >>> import numpy as np
    >>> from towhee import dc
    >>> df = dc['tensor']([np.array([i, j]) for i, j in zip(range(3), range(3))])
    >>> df.tensor_reshape['tensor', 'new_tensor'](shape = [2, 1]).map(lambda x: x.new_tensor.shape).to_list()
    [(2, 1), (2, 1), (2, 1)]

    >>> df.to_column()
    >>> df.tensor_reshape['tensor', 'new_new_tensor'](shape = [2, 1])['new_new_tensor'].map(lambda x: x.shape).to_list()
    [(2, 1), (2, 1), (2, 1)]
    """

    def __init__(self, shape):
        self._shape = shape

    def __call__(self, x):

        return x.reshape(self._shape)

    def __vcall__(self, x):
        self._shape.insert(0, -1)
        array = x.reshape(self._shape)

        return array


@register(name='builtin/tensor_random')
class tensor_random:
    """
    Return a random tensor filled with random numbers from [0.0, 1.0).

    Args:
        shape ([int], optional): tensor shape. Defaults to None.

    Returns:
        ndarray: output tensor.

    Examples:

    >>> import numpy as np
    >>> from towhee import dc
    >>> df = dc['a'](range(3))
    >>> df.tensor_random['a', 'b'](shape = [2, 1]).map(lambda x: x.b.shape).to_list()
    [(2, 1), (2, 1), (2, 1)]

    >>> df.to_column()
    >>> df.tensor_random['a', 'c'](shape = [2, 1])['c'].map(lambda x: x.shape).to_list()
    [(2, 1), (2, 1), (2, 1)]
    """

    def __init__(self, shape):
        self._shape = shape

    def __call__(self, _):
        import numpy as np

        return np.random.random(self._shape)

    def __vcall__(self, x):
        import numpy as np

        self._shape.insert(0, x.size)
        return np.random.random(self._shape)


@register(name='builtin/tensor_matmul')
class tensor_matmul:
    """
    Matrix multiplication.

    Examples:
    >>> import numpy as np
    >>> from towhee import DataFrame, Entity
    >>> from towhee.types.tensor_array import TensorArray
    >>> df = DataFrame([Entity(a = np.ones([2, 1]), b = np.ones([1, 2])) for _ in range(3)])
    >>> df.tensor_matmul[('a', 'b'), 'c']().to_list()
    [<Entity dict_keys(['a', 'b', 'c'])>, <Entity dict_keys(['a', 'b', 'c'])>, <Entity dict_keys(['a', 'b', 'c'])>]

    >>> df.to_column()
    >>> df.tensor_matmul[('a', 'b'), 'd']().to_list()
    [<EntityView dict_keys(['a', 'b', 'c', 'd'])>, <EntityView dict_keys(['a', 'b', 'c', 'd'])>, <EntityView dict_keys(['a', 'b', 'c', 'd'])>]df
    """

    def __init__(self, trans=None):
        self._trans = trans

    def __call__(self, x, y=None):
        import numpy as np

        if y is None and self._trans is not None:
            return np.matmul(x, self._trans)
        return np.matmul(x, y)

    def __vcall__(self, x, y=None):
        import numpy as np

        if y is None and self._trans is not None:
            return np.matmul(x, self._trans)
        return np.matmul(x, y)
