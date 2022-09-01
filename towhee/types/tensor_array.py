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

from typing import Tuple
import numpy as np

from towhee.utils.thirdparty.pyarrow_utils import pa


class _TensorArrayType(pa.PyExtensionType):
    """
    Tensor array type
    """

    def __init__(self, shape: Tuple[int, ...], dtype: pa.DataType):
        self._shape = shape
        self._ext_shape = [-1]
        for x in shape:
            self._ext_shape.append(x)
        super().__init__(pa.list_(dtype))

    @property
    def shape(self):
        return self._shape

    @property
    def ext_shape(self):
        return self._ext_shape

    def __reduce__(self):
        return _TensorArrayType, (self._shape, self.storage_type.value_type)

    def __arrow_ext_class__(self):
        return TensorArray


class TensorArray(pa.ExtensionArray):
    """
    Array for ndarrays
    """

    @classmethod
    def from_numpy(cls, data):
        """
        Create a TensroArray from numpy array.

        Args:
            data (`numpy.ndarray`):
                The ndarray to create the TensorArray from.

        Examples:
        >>> import numpy as np
        >>> from towhee.types import TensorArray
        >>> arr = TensorArray.from_numpy(np.arange(10).reshape([5,2]))
        >>> arr[0]
        array([0, 1])

        >>> arr = TensorArray.from_numpy(np.arange(36).reshape([6,2,3]))
        >>> arr[1]
        array([[ 6,  7,  8],
            [ 9, 10, 11]])
        >>> list(arr.chunks(5))[1]
        array([[[30, 31, 32],
                [33, 34, 35]]])
        """
        if isinstance(data, (list, tuple)):
            if np.isscalar(data[0]):
                return pa.array(data)
            data = np.stack(data, axis=0)
        if not isinstance(data, np.ndarray):
            raise ValueError('only support ndarray or list/tuple of ndarrays.')

        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        element_shape = data.shape[1:]
        num_items_per_element = np.prod(element_shape) if element_shape else 1

        data_array = pa.Array.from_buffers(pa.from_numpy_dtype(data.dtype),
                                           data.size,
                                           [None, pa.py_buffer(data)])
        offset_buffer = pa.py_buffer(
            np.int32(
                [i * num_items_per_element for i in range(data.shape[0] + 1)]))
        storage = pa.Array.from_buffers(
            pa.list_(pa.from_numpy_dtype(data.dtype)),
            data.shape[0],
            [None, offset_buffer],
            children=[data_array],
        )
        type_ = _TensorArrayType(element_shape,
                                 pa.from_numpy_dtype(data.dtype))
        return pa.ExtensionArray.from_storage(type_, storage)

    def __getitem__(self, index):
        if isinstance(index, slice):
            retval = super().__getitem__(index)
            storage = retval.storage
            return storage.flatten().to_numpy().reshape(self.type.ext_shape)
        retval = super().__getitem__(index)
        storage = retval.value.values
        return storage.to_numpy().reshape(self.type.shape)

    def to_numpy(self, zero_copy_only=True):
        """
        Create a numpy array from the TensorArray.

        Args:
            zero_copy_only (`bool`):
                Whether to create a copy of the array.

        Examples:
        >>> import numpy as np
        >>> from towhee.types import TensorArray
        >>> arr = TensorArray.from_numpy(np.arange(10).reshape([5,2]))
        >>> arr.to_numpy()
        array([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
        """
        return self.storage.flatten().to_numpy(zero_copy_only=zero_copy_only).reshape(self.type.ext_shape)

    def chunks(self, chunk_size=None):
        view = self.to_numpy()
        for i in range(0, len(self), chunk_size):
            yield view[i:i + chunk_size]

    def __iter__(self):
        return (self[i] for i in range(len(self)))
