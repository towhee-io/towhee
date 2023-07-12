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

from towhee.runtime.factory import HubOp


class ReRank:
    """
    Re-rank the search results based on relevance.
    """

    cross_encoder: HubOp = HubOp('towhee.rerank')
    """
    The Rerank operator is used to reorder the list of relevant documents for a query.
    It uses the `MS MARCO <https://www.sbert.net/docs/pretrained_cross-encoders.html#ms-marco>`_
    `Cross-Encoders <https://www.sbert.net/docs/pretrained_cross-encoders.html#ms-marco>`
    model to get the relevant scores and then reorders the documents.

    __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', threshold: float = 0.6):
        model_name(`str`):
            The model name of CrossEncoder, you can set it according to the
            `Model List <https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#models-performance>`_.
        threshold(`float`)
            The threshold for filtering with score
        device(`str`):
            Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.

    __call__(self, query: str, docs: List) -> List[str], List[float]
        query(`str`):
            The query content.
        docs(`Union[List[str], str]`):
            Sentences to check the correlation with the query content.
    Return(`List[str], List[float]`)
        docs and scores. The list of documents after rerank and the list of corresponding scores.

    Example:

    .. code-block:: python

        from towhee import ops, pipe, DataCollection

        p = (pipe.input('query', 'doc')
                .map(('query', 'doc'), ('doc', 'score'), ops.rerank.cross_encoder(threshold=0.9))
                .flat_map(('doc', 'score'), ('doc', 'score'), lambda x, y: [(i, j) for i, j in zip(x, y)])
                .output('query', 'doc', 'score')
            )

        DataCollection(p('What is Towhee?',
                        ['Towhee is Towhee is a cutting-edge framework to deal with unstructure data.',
                        'I do not know about towhee', 'Towhee has many powerful operators.',
                        'The weather is good' ])
                    ).show()
    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.rerank')(*args, **kwargs)

