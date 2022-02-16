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

from towhee import pipes

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


class TestPipelinePipes(unittest.TestCase):
    """
    tests for template build
    """

    def test_pipes(self):
        # pylint: disable=protected-access
        pipe = pipes.builtin.template_test(test_op='towhee/test_op')
        self.assertEqual(repr(pipe), render_result)


if __name__ == '__main__':
    unittest.main()
