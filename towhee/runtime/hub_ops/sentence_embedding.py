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


class SentenceEmbedding:
    """
    Sentence embedding is the extension of word or token embedding.
    Instead of generating a vector for each token or word,
    it represents semantic information of the whole text as a vector.
    """

    transformers: HubOp = HubOp('sentence_embedding.transformers')
    """
    `transformers <https://towhee.io/sentence-embedding/transformers>`_
    generates one embedding vector in ndarray for each input text.
    The embedding represents the semantic information of the whole input text as one vector.
    This operator is implemented with pre-trained models from Huggingface Transformers.

    Args:
        model_name(`str`):
            The model name in string, defaults to None.
            If None, the operator will be initialized without specified model.
            Supported model names: NLP transformers models listed in
            `Huggingface Models <https://huggingface.co/models>`_. Please note that only models
            listed in `supported_model_names` are tested. You can refer to
            `Towhee Pipeline <https://towhee.io/tasks/detail/pipeline/sentence-similarity>`_ for model performance.

        checkpoint_path(`str`):
            The path to local checkpoint, defaults to None.
            If None, the operator will download and load pretrained model by
            `model_name` from Huggingface transformers.

        tokenizer(`any`):
            The method to tokenize input text, defaults to None.
            If None, the operator will use default tokenizer by `model_name` from Huggingface transformers.

        device (`str`):
            Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('text')
                .map('text', 'vec',
                    ops.sentence_embedding.transformers(model_name='sentence-transformers/paraphrase-albert-small-v2'))
                .output('text', 'vec')
        )

        DataCollection(p('Hello, world.')).show()
    """

    sbert: HubOp = HubOp('sentence_embedding.sbert')
    """
    `sbert <https://towhee.io/sentence-embedding/sbert>`_ takes a sentence or a list of
    sentences in string as input. It generates an embedding vector in numpy.ndarray for each sentence,
    which captures the input sentence's core semantic elements.
    This operator is implemented with pre-trained models from Sentence Transformers.

    Args:
        model_name(`str`):
            The model name in string, defaults is all-MiniLM-L12-v2.
            Refer to `SBert Doc <https://www.sbert.net/docs/pretrained_models.html>`_.
            Please note that only models listed supported_model_names are tested. You can refer to
            `Towhee Pipeline <https://towhee.io/tasks/detail/pipeline/sentence-similarity>`_ for model performance.

        device (`str`):
            Device id: cpu/cuda:{GPUID}, if not set, will try to find an available GPU device.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('sentence')
            .map('sentence', 'vec', ops.sentence_embedding.sbert(model_name='all-MiniLM-L12-v2'))
            .output('sentence', 'vec')
        )

        DataCollection(p('This is a sentence.')).show()
    """

    openai: HubOp = HubOp('sentence_embedding.openai')
    """
    `openai <https://towhee.io/sentence-embedding/openai>`_ generates one embedding vector in ndarray for each input text.
    The embedding represents the semantic information of the whole input text as one vector.
    This operator is implemented with embedding models from OpenAI.
    Please note you need an OpenAI API key to access OpenAI.

    Args:
        model_name(`str`):
            The model name in string, defaults to 'text-embedding-ada-002'
        api_key(`str`):
            The OpenAI API key in string, defaults to None.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection

        p = (
            pipe.input('text')
                .map('text', 'vec',
                    ops.sentence_embedding.openai(model_name='text-embedding-ada-002', api_key=OPENAI_API_KEY))
                .output('text', 'vec')
        )

        DataCollection(p('Hello, world.')).show()
    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.sentence_embedding')(*args, **kwargs)
