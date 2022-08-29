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

from typing import Iterable, Tuple

from towhee.hparam import param_scope
from towhee.utils.log import engine_log
from ..entity import Entity


def _milvus_insert(iterable: Iterable, index: Tuple[str], collection, batch_size: int): # pragma: no cover
    # pylint: disable=import-outside-toplevel
    from towhee.utils.thirdparty.milvus_utils import Collection, MutationResult
    if isinstance(collection, str):
        collection = Collection(collection)
    primary_keys = []
    insert_count = 0
    first = True

    try:
        for entity in iterable:
            if first:
                if index is None:
                    index = tuple(entity.__dict__.keys())
                data = [[] for _ in range(len(index))]
                first = False

            for i in range(len(index)):
                data[i].append(getattr(entity, index[i]))
            if len(data[0]) == batch_size:
                mr = collection.insert(data)
                data = [[] for _ in range(len(index))]
                primary_keys += mr.primary_keys
                insert_count += mr.insert_count
                engine_log.info('Successfully inserted %d row data.', mr.insert_count)

        if len(data[0]) > 0:
            mr = collection.insert(data)
            primary_keys += mr.primary_keys
            insert_count += mr.insert_count
            engine_log.info('Successfully inserted %d row data.', mr.insert_count)

        e = Entity(insert_count=insert_count,
                   primary_keys=primary_keys,
                   delete_count=mr.delete_count,
                   upsert_count=mr.upsert_count,
                   timestamp=mr.timestamp)
        milvus_mr = MutationResult(e)
    except Exception as e:  # pylint: disable=broad-except
        engine_log.error('Error when insert data to milvus with %s.', e)
        raise e
    finally:
        collection.load()
    return milvus_mr


def _to_milvus_callback(self): # pragma: no cover
    # pylint: disable=consider-using-get
    def wrapper(_: str, index, *arg, **kws):
        batch_size = 1
        if index is not None and isinstance(index, str):
            index = (index,)

        if arg is not None and len(arg) == 1:
            collection, = arg
        elif arg is not None and len(arg) == 2:
            collection, batch_size = arg

        if 'collection' in kws:
            collection = kws['collection']
        if 'batch' in kws:
            batch_size = int(kws['batch'])

        dc_data = self
        if 'stream' in kws and not kws['stream']:
            dc_data = self.unstream()

        _ = _milvus_insert(dc_data, index, collection, batch_size)
        return dc_data
    return wrapper


class MilvusMixin: # pragma: no cover
    """
    Mixins for Milvus, such as loading data into Milvus collections. Note that the Milvus collection is created before loading the data.
    Refer to https://milvus.io/docs/v2.0.x/create_collection.md.

    Args:
        collection (`Union[str, Collection]`):
            The collection name or pymilvus.Collection in Milvus.
        batch (`str`):
            The batch size to load into Milvus, defaults to 1.
        stream (`bool`, optional):
            Whther the stream mode with `to_milvus`, defaults to True.

    Returns:
        The DataCollection.

    Examples:

    .. note::
        The shape of embedding vector refer to https://towhee.io/image-embedding/timm. And the dimension of the "test" collection should
        be the same as it.

    >>> import towhee
    >>> mr = ( #doctest: +SKIP
    ...     towhee.glob['path']('./*.jpg')
    ...           .image_decode['path', 'img']()
    ...           .image_embedding.timm['img', 'vec'](model_name='resnet50')
    ...           .to_milvus['vec'](collection='test', batch=1000)
    ... )
    """
    def __init__(self):
        super().__init__()
        self.to_milvus = param_scope().dispatch(_to_milvus_callback(self))
