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

from towhee import Build, Inject
from towhee.utils.yaml_utils import load_yaml, dump_yaml

render_result = """
name: test_pipeline
variables:
    test_op: 'filip-halt/timm-image-embedding'
    model_name: 'efficientnet_b3'
operators:
- name: embedding_model
  function: 'towhee/test_op'
  tag: main
  init_args:
    model_name: efficientnet_b3
  inputs:
  - df: image
    name: image
    col: 0
  outputs:
  - df: embedding
  iter_info:
    type: map
dataframes:
- name: embedding
  columns:
  - name: feature_vector
    vtype: numpy.ndarray
""".strip()


class TestTemplateBuild(unittest.TestCase):
    """
    tests for template build
    """
    def test_template_build(self):
        pipe = Build(test_op='towhee/test_op').pipeline('builtin/template_test')
        self.assertEqual(repr(pipe), render_result)


class TestTemplateInject(unittest.TestCase):
    """
    tests for template inject
    """
    def test_template_inject(self):
        pipe = Inject(embedding_model=dict(function='towhee/test_op')).pipeline('builtin/template_test')
        self.assertEqual(dump_yaml(load_yaml(repr(pipe))), dump_yaml(load_yaml(render_result)))


if __name__ == '__main__':
    unittest.main()
