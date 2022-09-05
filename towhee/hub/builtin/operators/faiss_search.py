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

from towhee.engine import register

# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


@register(name='builtin/faiss_search')
class faiss_search:
    """
    Search for embedding vectors in Faiss. Note that the index has data before searching,
    refer to DataCollection Mixin `to_faiss`.

    Args:
        findex (`str` or `faiss.INDEX`):
            The path to faiss index file(defaults to './index.bin') or faiss index.
        kwargs
            The kwargs with index.search, refer to https://github.com/facebookresearch/faiss/wiki. And the parameter `k` defaults to 10.

    Examples:
    ```
    import towhee
    res = (
       towhee.glob['path']('./*.jpg')
              .image_decode['path', 'img']()
              .image_embedding.timm['img', 'vec'](model_name='resnet50')
              .faiss_search['vec', 'results'](findex='./faiss/faiss.index')
              .to_list()
    )
    ```
    """
    def __init__(self, findex, **kwargs):
        from towhee.utils.thirdparty.faiss_utils import KVStorage, faiss
        self.faiss_index = findex
        self.kwargs = kwargs
        self.kv_storage = None
        if isinstance(findex, str):
            kv_file = findex.strip('./').replace('.', '_kv.')
            index_file = Path(findex)
            self.faiss_index = faiss.read_index(str(index_file))
            if Path(kv_file).exists():
                self.kv_storage = KVStorage(kv_file)
        if 'k' not in self.kwargs:
            self.kwargs['k'] = 10

    def __call__(self, query: list):
        from towhee.functional.entity import Entity

        query = np.array([query])
        scores, ids = self.faiss_index.search(query, **self.kwargs)

        ids = ids[0].tolist()
        result = []
        for i in range(len(ids)):
            if self.kv_storage is not None:
                k = self.kv_storage.get(ids[i])
            else:
                k = ids[i]
            result.append(Entity(**{'key': k, 'score': scores[0][i]}))
        return result
