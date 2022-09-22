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

# pylint: disable=import-outside-toplevel
# pylint: disable=protected-access
# pylint: disable=consider-using-get
from io import BytesIO
from typing import Iterable

import numpy as np
from towhee.hparam import param_scope


def _insert_leveldb_callback(self):

    def wrapper(_, index, *arg, **kws):
        from towhee.utils.thirdparty.plyvel_utils import plyvel
        path = None
        if arg is not None and len(arg) == 1:
            path = arg[0]
        if 'path' in kws:
            path = kws['path']

        db = plyvel.DB(path, create_if_missing=True)

        dc_data = self
        if self.is_stream:
            dc_data = self.unstream()

        for i in dc_data._iterable:
            key = getattr(i, index[0][0])
            val = getattr(i, index[0][1])

            if isinstance(key, str) or not isinstance(key, Iterable):
                key = [key]

            if isinstance(val, np.ndarray):
                np_bytes = BytesIO()
                np.save(np_bytes, val, allow_pickle=True)
                val = np_bytes.getvalue()
            else:
                val = str(val).encode('utf-8')

            for k in key:
                db.put(str(k).encode('utf-8'), val)

        db.close()
        return dc_data

    return wrapper


def _from_leveldb_callback(self):

    def wrapper(_, index, *arg, **kws):
        from towhee.utils.thirdparty.plyvel_utils import plyvel
        path = None
        is_ndarray = False

        if arg is not None and len(arg) == 1:
            path = arg[0]
        elif arg is not None and len(arg) == 2:
            path = arg[0]
            is_ndarray = arg[1]

        if 'path' in kws:
            path = kws['path']
        if 'is_ndarray' in kws:
            is_ndarray = kws['is_ndarray']

        db = plyvel.DB(path, create_if_missing=True)

        dc_data = self
        if self.is_stream:
            dc_data = self.unstream()

        for i in dc_data._iterable:
            key = getattr(i, index[0])
            if isinstance(key, str) or not isinstance(key, Iterable):
                val = db.get(str(key).encode('utf-8'))
                if not is_ndarray:
                    val.decode('utf-8')
                else:
                    val = BytesIO(val)
                    val = np.load(val, allow_pickle=True)

                setattr(i, index[1], val)
            else:
                vals = []
                for k in key:
                    val = db.get(str(k).encode('utf-8'))
                    if not is_ndarray:
                        vals.append(val.decode('utf-8'))
                    else:
                        val = BytesIO(val)
                        val = np.load(val, allow_pickle=True)
                        vals.append(val)
                setattr(i, index[1], vals)

        db.close()
        return dc_data

    return wrapper


class KVStorageMixin: # pragma: no cover
    """
    Mixin for kv storage.
    """
    def __init__(self):
        super().__init__()
        self.insert_leveldb = param_scope().dispatch(_insert_leveldb_callback(self))
        self.from_leveldb = param_scope().dispatch(_from_leveldb_callback(self))
