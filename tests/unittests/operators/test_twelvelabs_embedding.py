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
import unittest
from unittest import mock

from towhee import ops
from towhee.operator.twelvelabs_embedding import TwelveLabsEmbedding


def _fake_embedding_response(dim=512):
    segment = mock.MagicMock()
    segment.float_ = [0.0] * dim
    response = mock.MagicMock()
    response.text_embedding.segments = [segment]
    return response


class TestTwelveLabsEmbedding(unittest.TestCase):
    """
    Unit tests for the TwelveLabs Marengo embedding operator. These do not hit
    the network; the SDK client is mocked.
    """

    def test_requires_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                TwelveLabsEmbedding()

    def test_invalid_modality(self):
        with self.assertRaises(ValueError):
            TwelveLabsEmbedding(modality='nonsense', api_key='dummy')

    def test_registered_with_ops(self):
        op = ops.twelvelabs_embedding(api_key='dummy')
        self.assertEqual(op.name, 'twelvelabs-embedding')

    def test_text_embedding(self):
        op = TwelveLabsEmbedding(api_key='dummy')
        fake_client = mock.MagicMock()
        fake_client.embed.create.return_value = _fake_embedding_response()
        op._client = fake_client  # pylint: disable=protected-access

        vec = op('a person playing guitar')

        self.assertEqual(len(vec), 512)
        fake_client.embed.create.assert_called_once_with(
            model_name='marengo3.0', text='a person playing guitar'
        )

    def test_image_url_routing(self):
        op = TwelveLabsEmbedding(modality='image', api_key='dummy')
        fake_client = mock.MagicMock()
        resp = mock.MagicMock()
        resp.image_embedding.segments = [mock.MagicMock(float_=[0.0] * 512)]
        fake_client.embed.create.return_value = resp
        op._client = fake_client  # pylint: disable=protected-access

        op('https://example.com/cat.jpg')

        fake_client.embed.create.assert_called_once_with(
            model_name='marengo3.0', image_url='https://example.com/cat.jpg'
        )


@unittest.skipUnless(
    os.environ.get('TWELVELABS_API_KEY'),
    'TWELVELABS_API_KEY not set; skipping live TwelveLabs API test.',
)
class TestTwelveLabsEmbeddingLive(unittest.TestCase):
    """
    Live smoke test against the TwelveLabs API. Only runs when
    TWELVELABS_API_KEY is set in the environment.
    """

    def test_live_text_embedding(self):
        op = ops.twelvelabs_embedding()
        vec = op('a person playing guitar')
        self.assertEqual(len(vec), 512)
        self.assertTrue(any(abs(x) > 0 for x in vec))


if __name__ == '__main__':
    unittest.main()
