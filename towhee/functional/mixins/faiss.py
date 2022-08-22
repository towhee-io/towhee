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

import numpy as np
from pathlib import Path
from typing import Iterable, Tuple

from towhee.hparam import param_scope
from towhee.utils.log import engine_log

# pylint: disable=consider-using-get
# pylint: disable=import-outside-toplevel


def get_faiss_index(findex, dim, index_str, metric):
    from towhee.utils.thirdparty.faiss_utils import faiss
    if not isinstance(findex, Path):
        return findex
    elif not findex.exists():
        findex.parent.mkdir(exist_ok=True, parents=True)
        return faiss.index_factory(dim, index_str, metric)
    else:
        return faiss.read_index(str(findex))


def _faiss_insert(iterable: Iterable, column: Tuple[str], findex, string, metric): # pragma: no cover
    from towhee.utils.thirdparty.faiss_utils import KVStorage, faiss
    if isinstance(findex, str):
        findex = Path(findex)
        kv_file = Path(str(findex).replace('.', '_kv.'))
    else:
        kv_file = Path('./index_kv.bin')
    ids = []
    vecs = []
    first = True
    for it in iterable:
        if first:
            enable_kv = False
            if not isinstance(getattr(it, column[0]), int):
                kv_storage = KVStorage(kv_file)
                enable_kv = True
            dim = len(getattr(it, column[1]))
            faiss_index = get_faiss_index(findex, dim, string, metric)
            first = False

        vec = getattr(it, column[1])
        vecs.append(vec)
        if enable_kv:
            k = getattr(it, column[0])
            vid = abs(hash(k)) % (10 ** 8)
            kv_storage.add(vid, k)
        else:
            vid = getattr(it, column[0])
        ids.append(vid)

    if first:
        engine_log.error('There is no data to insert into Faiss.')
        raise KeyError('There is no data to insert into Faiss.')

    faiss_index.add_with_ids(np.array(vecs), np.array(ids).astype(np.int64))
    if not isinstance(findex, Path):
        findex = Path('./index.bin')
    faiss.write_index(faiss_index, str(findex))
    if enable_kv:
        kv_storage.dump()

    return str(findex), str(kv_file) if enable_kv else None


def _to_faiss_callback(self): # pragma: no cover
    def wrapper(_: str, index, *arg, **kws):
        from towhee.utils.thirdparty.faiss_utils import faiss
        findex = './index.bin'
        string = 'IDMap,Flat'
        metric = faiss.METRIC_L2
        if index is None or len(index) != 2:
            engine_log.error('Make sure you have passed in two data(such as `ids` and `vectors`).')
            raise KeyError('Make sure you have passed in two data(such as `ids` and `vectors`).')

        if arg is not None and len(arg) == 1:
            findex, = arg
        elif arg is not None and len(arg) == 2:
            findex, string = arg
        elif arg is not None and len(arg) == 3:
            findex, string, metric = arg
        elif arg is not None and len(arg) > 3:
            engine_log.error('There are three parameters: findex(defaults to \'./index.bin\'), '
                             'string(defaults to \'IDMap,Flat\') and metric(defaults to `faiss.METRIC_L2`.)')
            raise KeyError('There are three parameters: findex(defaults to \'./index.bin\'), '
                           'string(defaults to \'IDMap,Flat\') and metric(defaults to `faiss.METRIC_L2`.)')

        if 'findex' in kws:
            findex = kws['findex']
        if 'string' in kws:
            string = int(kws['string'])
        if 'metric' in kws:
            metric = int(kws['metric'])

        dc_data = self
        if 'stream' in kws and not kws['stream']:
            dc_data = self.unstream()

        _, _ = _faiss_insert(dc_data, index, findex, string, metric)
        return dc_data
    return wrapper


class FaissMixin: # pragma: no cover
    """
    Mixins for Faiss, such as loading data into Faiss. And `ids` and `vectors` need to be passed as index. If ids is a string, KV storage will be
    started, and the kv data  will be saved to the specified directory as "kv.bin".

    Args:
        findex (`str` or `faiss.INDEX`, optional):
            The path to faiss index file(defaults to './index.bin') or faiss index.
        string (`str`, optional):
             A string to produce a composite Faiss index, which is the same parameter in `faiss.index_factor`, defaults to 'IDMap,Flat',
             and you can refer to https://github.com/facebookresearch/faiss/wiki/The-index-factory.
        metric (`faiss.METRIC`, optional):
            The metric for Faiss index, defaults to faiss.METRIC_L2.

    Returns:
        A DC, and will save the Faiss index file and kv file(if ids is string).

    Examples:

    .. note::
        Please make sure the path to `index_file` is authorized, and it will write the Faiss index file and kv file(if ids is string).

    >>> import towhee
    >>> dc = ( #doctest: +SKIP
    ...     towhee.glob['path']('./*.jpg')
    ...           .image_decode['path', 'img']()
    ...           .image_embedding.timm['img', 'vec'](model_name='resnet50')
    ...           .to_faiss['path', 'vec'](findex='./faiss/faiss.index')
    ... )
    """
    def __init__(self):
        super().__init__()
        self.to_faiss = param_scope().dispatch(_to_faiss_callback(self))
