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

import unittest

from towhee.utils.repo_normalize import RepoNormalize

rn1 = RepoNormalize('resnet-image-embedding?ref=my-tag&model=resnet50#operator')
rn2 = RepoNormalize('test/resnet_image_embedding[pytorch]')
rn3 = RepoNormalize('resnet-image-embedding')
rn4 = RepoNormalize('resnet_image_embedding')


class TestRepoNormalize(unittest.TestCase):
    """
    Unittest for repo normalize.
    """

    def test_parse_uri(self):
        result = rn1.parse_uri()
        self.assertEqual(result.full_uri, 'https://towhee.io/towhee/resnet-image-embedding?ref=my-tag&model=resnet50#operator')
        self.assertEqual(result.author, 'towhee')
        self.assertEqual(result.repo, 'resnet-image-embedding')
        self.assertEqual(result.ref, 'my-tag')
        self.assertEqual(result.repo_type, 'operator')
        self.assertEqual(result.norm_repo, 'resnet-image-embedding')
        self.assertEqual(result.module_name, 'resnet_image_embedding')
        self.assertEqual(result.class_name, 'ResnetImageEmbedding')
        self.assertEqual(result.scheme, 'https')
        self.assertEqual(result.netloc, 'towhee.io')
        self.assertEqual(result.query, {'model': 'resnet50'})

    def test_parse_uri_2(self):
        result = rn2.parse_uri()
        self.assertEqual(result.full_uri, 'https://towhee.io/test/resnet_image_embedding-pytorch?&ref=main')
        self.assertEqual(result.author, 'test')
        self.assertEqual(result.repo, 'resnet_image_embedding-pytorch')
        self.assertEqual(result.ref, 'main')
        self.assertEqual(result.repo_type, '')
        self.assertEqual(result.norm_repo, 'resnet-image-embedding-pytorch')
        self.assertEqual(result.module_name, 'resnet_image_embedding_pytorch')
        self.assertEqual(result.class_name, 'ResnetImageEmbeddingPytorch')
        self.assertEqual(result.scheme, 'https')
        self.assertEqual(result.netloc, 'towhee.io')
        self.assertEqual(result.query, {})

    def test_check_uri(self):
        self.assertTrue(rn1.check_uri())
        self.assertTrue(rn3.check_uri())
        with self.assertRaises(ValueError):
            rn4.check_uri()
        self.assertTrue(RepoNormalize('https://towhee.io/towhee/audio-embedding').check_uri())
        self.assertFalse(RepoNormalize('towhee/test/audio-embedding').check_uri())

    def test_mapping(self):
        with self.assertRaises(ValueError):
            RepoNormalize.mapping('resnet_image_embedding[pytorch')

    def test_get_op(self):
        result = RepoNormalize.get_op('test_op')
        self.assertEqual(result.repo, 'test-op')
        self.assertEqual(result.py_file, 'test_op.py')
        self.assertEqual(result.yaml_file, 'test_op.yaml')
        self.assertEqual(result.class_name, 'TestOp')

    def test_get_pipeline(self):
        result = RepoNormalize.get_pipeline('test_pipeline')
        self.assertEqual(result.repo, 'test-pipeline')
        self.assertEqual(result.yaml_file, 'test_pipeline.yaml')


if __name__ == '__main__':
    unittest.main()
