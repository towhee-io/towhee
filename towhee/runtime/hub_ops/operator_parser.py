# Copyright 2023 Zilliz. All rights reserved.
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

from towhee.runtime.factory import ops_parse, _OperatorWrapper
from .image_embedding import ImageEmbedding
from .image_decode import ImageDecode
from .sentence_embedding import SentenceEmbedding
from .data_source import DataSource
from .ann_insert import AnnInsert


class Ops:
    """
    Class to loading operator with _OperatorWrapper.
    An operator is usually referred to with its full name: namespace/name.

    Examples:
        >>> from towhee import ops
        >>> op = ops.towhee.image_decode()
        >>> img = op('./towhee_logo.png')

        We can also specify the version of the operator on the hub via the `revision` method:

        >>> op = ops.towhee.image_decode()

        And the `latest` method is used to update the current version of the operator to the latest:

        >>> op = ops.towhee.image_decode().latest()
    """

    image_embedding: ImageEmbedding = ImageEmbedding()
    """
     `image_embedding <https://towhee.io/tasks/detail/operator?field_name=Computer-Vision&task_name=Image-Embedding>`_
      is a task that attempts to comprehend an entire image as a whole
      and encode the image's semantics into a real vector.
      It is a fundamental task type that can be used in a variety of applications,
      including but not limited to reverse image search and image deduplication.
    """

    image_decode: ImageDecode = ImageDecode()
    """
    `image_decode <https://towhee.io/tasks/detail/operator?field_name=Computer-Vision&task_name=Image-Decode>`_
    operators convert an encoded image back to its uncompressed format.
    In most cases, image decoding is the first step of an image processing pipeline.
    """

    sentence_embedding: SentenceEmbedding = SentenceEmbedding()
    """
    `sentence_embedding <https://towhee.io/tasks/detail/operator?field_name=Natural-Language-Processing&task_name=Sentence-Embedding>`_
    is the extension of word or token embedding. Instead of generating a vector for each token or word,
    it represents semantic information of the whole text as a vector.
    """

    data_source: DataSource = DataSource()
    """
    `data_source <https://towhee.io/data-source?index=1&size=30&type=2>`_ load data from many different sources.
    Work with `DataLoader <https://towhee.readthedocs.io/en/latest/data_source/data_source.html>`_
    """

    ann_insert: AnnInsert = AnnInsert()
    """
    The ANN Insert Operator is used to insert embeddings and create ANN indexes for fast similarity searches.
    """

    @classmethod
    def __getattr__(cls, name):
        @ops_parse
        def wrapper(name, *args, **kws):
            return _OperatorWrapper.callback(name, *args, **kws)

        return getattr(wrapper, name)
