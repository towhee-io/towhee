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
import os
from pathlib import Path

from towhee.utils.yaml_utils import load_yaml, dump_yaml
import ruamel.yaml

PIPELINE_PATH = Path(__file__).parent.parent.resolve() / 'test_util' / 'resnet50_embedding' / 'resnet50_embedding.yaml'
cache_path = Path(__file__).parent.parent.resolve() / 'test_cache' / 'test_yaml.yaml'


class TestYamlUtils(unittest.TestCase):
    """
    Unit test for yaml utils.
    """
    def test_load_yaml(self):
        with open(PIPELINE_PATH, 'r', encoding='utf-8') as input_file:
            data = load_yaml(stream=input_file, typ='safe')
            self.assertIsInstance(data, dict)
            self.assertIn('name', data.keys())
            self.assertIn('operators', data.keys())
            self.assertIn('dataframes', data.keys())

        with open(PIPELINE_PATH, 'r', encoding='utf-8') as input_file:
            data_1 = load_yaml(stream=input_file, typ=None)
            self.assertIsInstance(data_1, ruamel.yaml.comments.CommentedMap)

        src = '{A: a, B: b, C: c}'
        data = load_yaml(stream=src, typ='safe')
        self.assertIsInstance(data, dict)
        self.assertIn('A', data.keys())
        self.assertIn('B', data.keys())
        self.assertIn('C', data.keys())

    def test_dump_yaml(self):
        data = {'A': 'a', 'B': 'b', 'C': 'c'}
        if cache_path.is_file():
            os.remove(cache_path)
        with open(cache_path, 'w', encoding='utf-8') as output_file:
            dump_yaml(data=data, stream=output_file)
        self.assertTrue(cache_path.is_file())
        self.assertEqual(cache_path.suffix, '.yaml')
        os.remove(cache_path)

        string = dump_yaml(data=data)
        self.assertIsInstance(string, str)
