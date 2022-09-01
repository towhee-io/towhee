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

from typing import Union, List
from towhee.hparam import param_scope


class ConfigMixin:
    """
    Mixin to config DC, such as set the `parallel`, `chunksize`, `jit` and `format_priority`.

    Examples:

        >>> import towhee
        >>> dc = towhee.dc['a'](range(20))
        >>> dc = dc.set_chunksize(10)
        >>> dc = dc.set_parallel(2)
        >>> dc = dc.set_jit('numba')
        >>> dc.get_config()
        {'parallel': 2, 'chunksize': 10, 'jit': 'numba', 'format_priority': None}
        >>> dc1 = towhee.dc([1,2,3]).config(jit='numba')
        >>> dc2 = towhee.dc['a'](range(40)).config(parallel=2, chunksize=20)
        >>> dc1.get_config()
        {'parallel': None, 'chunksize': None, 'jit': 'numba', 'format_priority': None}
        >>> dc2.get_config()
        {'parallel': 2, 'chunksize': 20, 'jit': None, 'format_priority': None}
        >>> dc3 = towhee.dc['a'](range(10)).config(format_priority=['tensorrt', 'onnx'])
        >>> dc3.get_config()
        {'parallel': None, 'chunksize': None, 'jit': None, 'format_priority': ['tensorrt', 'onnx']}

        >>> import towhee
        >>> dc = towhee.dc['a'](range(20))
        >>> dc = dc.set_chunksize(10)
        >>> dc = dc.set_parallel(2)
        >>> dc = dc.set_jit('numba')
        >>> dc.get_pipeline_config()
        {'parallel': 2, 'chunksize': 10, 'jit': 'numba', 'format_priority': None}
        >>> dc1 = towhee.dc([1,2,3]).pipeline_config(jit='numba')
        >>> dc2 = towhee.dc['a'](range(40)).pipeline_config(parallel=2, chunksize=20)
        >>> dc1.get_pipeline_config()
        {'parallel': None, 'chunksize': None, 'jit': 'numba', 'format_priority': None}
        >>> dc2.get_pipeline_config()
        {'parallel': 2, 'chunksize': 20, 'jit': None, 'format_priority': None}
        >>> dc3 = towhee.dc['a'](range(10)).pipeline_config(format_priority=['tensorrt', 'onnx'])
        >>> dc3.get_pipeline_config()
        {'parallel': None, 'chunksize': None, 'jit': None, 'format_priority': ['tensorrt', 'onnx']}
    """

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_config'):
            self._config = parent._config
        else:
            self._config = None
        if parent is None or not hasattr(parent, '_num_worker'):
            self._num_worker = None
        if parent is None or not hasattr(parent, '_chunksize'):
            self._chunksize = None
        if parent is None or not hasattr(parent, '_jit'):
            self._jit = None
        if parent is None or not hasattr(parent, '_format_priority'):
            self._format_priority = None

    def config(self, parallel: int = None, chunksize: int = None, jit: Union[str, dict] = None, format_priority: List[str] = None):
        """
        Set the parameters for the DC.

        Args:
            parallel (int, optional): Set the number of parallel execution for the following calls, defaults to None.
            chunksize (int, optional): Set the chunk size for arrow, defaults to None.
            jit (Union[str, dict], optional): Can be set to "numba", this mode will speed up the Operator's function,
                but it may also need to return to python mode due to JIT failure, which will take longer, so please
                set it carefully, defaults to None.
            format_priority (List[str], optional): The priority list of formats, defaults to None.

        Returns:
            DataCollection: Self.
        """
        dc = self
        if jit is not None:
            dc = dc.set_jit(compiler=jit)
        if parallel is not None:
            dc = dc.set_parallel(num_worker=parallel)
        if chunksize is not None:
            dc = dc.set_chunksize(chunksize=chunksize)
        if format_priority is not None:
            dc = dc.set_format_priority(format_priority=format_priority)
        return dc

    def get_config(self):
        """
        Return the config of the DC, including parameters such as `parallel`, `chunksize`, `jit` and `format_priority`.

        Returns:
            dict: A dict of config parameters.
        """
        self._config = {}

        if hasattr(self, '_num_worker'):
            self._config['parallel'] = self._num_worker
        if hasattr(self, '_chunksize'):
            self._config['chunksize'] = self._chunksize
        if hasattr(self, '_jit'):
            self._config['jit'] = self._jit
        if hasattr(self, '_format_priority'):
            self._config['format_priority'] = self._format_priority
        return self._config

    def pipeline_config(self, parallel: int = None, chunksize: int = None, jit: Union[str, dict] = None, format_priority: List[str] = None):
        """
        Set the parameters in DC.

        Args:
            parallel (int, optional): Set the number of parallel executions for the following calls, defaults to None.
            chunksize (int, optional): Set the chunk size for arrow, defaults to None.
            jit (Union[str, dict], optional): Can be set to "numba", this mode will speed up the Operator's function,
                but it may also need to return to python mode due to JIT failure, which will take longer, so please
                set it carefully, defaults to None.
            format_priority (List[str], optional): The priority list of format, defaults to None.

        Returns:
            DataCollection: Self
        """
        dc = self
        if jit is not None:
            dc = dc.set_jit(compiler=jit)
        if parallel is not None:
            dc = dc.set_parallel(num_worker=parallel)
        if chunksize is not None:
            dc = dc.set_chunksize(chunksize=chunksize)
        if format_priority is not None:
            dc = dc.set_format_priority(format_priority=format_priority)
        return dc

    def get_pipeline_config(self):
        """
        Return the config of the DC, including parameters such as `parallel`, `chunksize`, `jit` and `format_priority`.

        Returns:
            dict: A dict of config parameters.
        """
        self._pipeline_config = {}

        if hasattr(self, '_num_worker'):
            self._pipeline_config['parallel'] = self._num_worker
        if hasattr(self, '_chunksize'):
            self._pipeline_config['chunksize'] = self._chunksize
        if hasattr(self, '_jit'):
            self._pipeline_config['jit'] = self._jit
        if hasattr(self, '_format_priority'):
            self._pipeline_config['format_priority'] = self._format_priority
        return self._pipeline_config
