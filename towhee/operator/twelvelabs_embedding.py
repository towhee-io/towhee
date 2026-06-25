# Copyright 2024 Zilliz. All rights reserved.
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

import os
from typing import List, Union

from towhee.operator import PyOperator
from towhee.runtime import register


@register(name='twelvelabs_embedding')
class TwelveLabsEmbedding(PyOperator):
    """
    Generate multimodal embeddings with the
    `TwelveLabs <https://twelvelabs.io>`_ Marengo model.

    Marengo maps text, images, audio and video into a single 512-dimensional
    embedding space, which makes it convenient for cross-modal retrieval
    pipelines (e.g. text-to-video search). Video is embedded server-side as an
    asynchronous task; text/image/audio are embedded synchronously.

    You need a TwelveLabs API key to use this operator. Pass it via the
    ``api_key`` argument or the ``TWELVELABS_API_KEY`` environment variable.
    A free key with a generous free tier is available at https://twelvelabs.io.

    Args:
        modality (`str`):
            The input modality, one of ``text``, ``image``, ``audio`` or
            ``video``. Defaults to ``text``.
        model_name (`str`):
            The Marengo model to use. Defaults to ``marengo3.0``.
        api_key (`str`):
            The TwelveLabs API key. If not given, the ``TWELVELABS_API_KEY``
            environment variable is used instead.

    Example:

    .. code-block:: python

        from towhee import pipe, ops, DataCollection
        # the import below registers the operator with `ops`
        import towhee.operator.twelvelabs_embedding  # noqa: F401

        p = (
            pipe.input('text')
                .map('text', 'vec', ops.twelvelabs_embedding())
                .output('text', 'vec')
        )

        DataCollection(p('a person playing guitar')).show()
    """

    def __init__(self, modality: str = 'text', model_name: str = 'marengo3.0', api_key: str = None):
        super().__init__()
        modality = modality.lower()
        if modality not in ('text', 'image', 'audio', 'video'):
            raise ValueError(
                f'Unsupported modality "{modality}", expected one of '
                'text/image/audio/video.'
            )
        self._modality = modality
        self._model_name = model_name
        self._api_key = api_key or os.environ.get('TWELVELABS_API_KEY')
        if not self._api_key:
            raise ValueError(
                'A TwelveLabs API key is required. Pass it via the `api_key` '
                'argument or set the `TWELVELABS_API_KEY` environment variable. '
                'Get a free key at https://twelvelabs.io.'
            )
        self._client = None

    @property
    def client(self):
        # Lazily create the SDK client so importing the operator does not
        # require the `twelvelabs` package to be installed.
        if self._client is None:
            try:
                from twelvelabs import TwelveLabs  # pylint: disable=import-outside-toplevel
            except ImportError as exc:
                raise ImportError(
                    'The `twelvelabs` package is required for the '
                    'twelvelabs_embedding operator. Install it with '
                    '`pip install twelvelabs`.'
                ) from exc
            self._client = TwelveLabs(api_key=self._api_key)
        return self._client

    def _embed_video(self, data: str) -> List[List[float]]:
        kwargs = {'model_name': self._model_name}
        if data.startswith(('http://', 'https://')):
            kwargs['video_url'] = data
        else:
            kwargs['video_file'] = data
        task = self.client.embed.tasks.create(**kwargs)
        self.client.embed.tasks.wait_for_done(task_id=task.id)
        result = self.client.embed.tasks.retrieve(task.id)
        segments = result.video_embedding.segments
        return [list(seg.float_) for seg in segments]

    def _embed_other(self, data: str) -> List[float]:
        kwargs = {'model_name': self._model_name}
        if self._modality == 'text':
            kwargs['text'] = data
        elif self._modality == 'image':
            if data.startswith(('http://', 'https://')):
                kwargs['image_url'] = data
            else:
                kwargs['image_file'] = data
        else:  # audio
            if data.startswith(('http://', 'https://')):
                kwargs['audio_url'] = data
            else:
                kwargs['audio_file'] = data
        result = self.client.embed.create(**kwargs)
        embedding = getattr(result, f'{self._modality}_embedding')
        return list(embedding.segments[0].float_)

    def __call__(self, data: str) -> Union[List[float], List[List[float]]]:
        """
        Embed a single input.

        Args:
            data (`str`):
                The input to embed. For ``text`` this is the raw string; for
                ``image``/``audio``/``video`` it is a URL or a local file path.

        Returns:
            For ``text``/``image``/``audio`` a single 512-dim vector
            (`List[float]`). For ``video`` a list of per-segment 512-dim
            vectors (`List[List[float]]`), one per clip.
        """
        if self._modality == 'video':
            return self._embed_video(data)
        return self._embed_other(data)
