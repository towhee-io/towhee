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

# pylint: disable=bare-except

class VectorizedExecution:
    """
    Vectorized execution of an operator on Arrow tables.
    """

    def __vcall__(self, *arg, **kws):
        self.__check_init__()
        # col-based computing supported
        if hasattr(self._op, '__vcall__'):
            return self._op.__vcall__(*arg, **kws)
        elif len(arg) == 1:
            res = [self._op(x) for x in arg[0]]
            if isinstance(self._index[1], tuple):
                return tuple(list(i) for i in zip(*res))
            else:
                return res
        else:
            res = [self._op(*x) for x in zip(*arg)]
            if isinstance(self._index[1], tuple):
                return tuple(list(i) for i in zip(*res))
            else:
                return res

        # if bool(self._index):
        #     args = []
        #     if isinstance(self._index[0], tuple):
        #         # Multi inputs.
        #         for col in self._index[0]:
        #             buffer = arg[0][col].chunk(0).buffers()[-1]
        #             shape = [-1,
        #                      *arg[0][col].chunk(0).type.shape] if isinstance(
        #                          arg[0][col].chunk(0),
        #                          TensorArray) else [len(arg[0][col]), -1]
        #             dtype = arg[0][col].chunk(0).type.storage_type.value_type if isinstance(arg[0][col].chunk(0), TensorArray) \
        #                 else arg[0][col].type.value_type if hasattr(arg[0][col].type, 'value_type') \
        #                 else arg[0][col].type
        #             dtype = dtype.to_pandas_dtype()
        #             args.append(
        #                 np.frombuffer(buffer=buffer,
        #                               dtype=dtype).reshape(shape))
        #     else:
        #         # Single input.
        #         col = self._index[0]
        #         buffer = arg[0][col].chunk(0).buffers()[-1]
        #         shape = [-1, *arg[0][col].chunk(0).type.shape] if isinstance(
        #             arg[0][col].chunk(0),
        #             TensorArray) else [len(arg[0][col]), -1]
        #         dtype = arg[0][col].chunk(0).type.storage_type.value_type if isinstance(arg[0][col].chunk(0), TensorArray) \
        #             else arg[0][col].type.value_type if hasattr(arg[0][col].type, 'value_type') \
        #             else arg[0][col].type
        #         dtype = dtype.to_pandas_dtype()
        #         args.append(
        #             np.frombuffer(buffer=buffer, dtype=dtype).reshape(shape))
        #         # args.append(arg[0][col].chunks[0].as_numpy())

        # if hasattr(self._op, '__vcall__'):
        #     res = self._op.__vcall__(*args, **kws)
        #     if isinstance(res, tuple):
        #         # Mulit outputs.
        #         arrs = [TensorArray.from_numpy(x) for x in res]
        #         table = arg[0]
        #         for i, j in zip(self._index[1], arrs):
        #             table = table.append_column(i, j)
        #         return table
        #     else:
        #         # Single input.
        #         arr = TensorArray.from_numpy(res)
        #         table = arg[0].append_column(self._index[1], arr)
        #         return table
        # else:
        #     self.__call__(*args, **kws)
