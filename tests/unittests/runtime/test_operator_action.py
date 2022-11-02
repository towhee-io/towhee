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

from towhee.runtime.operator_manager.operator_action import OperatorAction


# pylint: disable=protected-access
class TestOperatorAction(unittest.TestCase):
    """
    Test OperatorAction
    """

    def test_hub(self):
        op_action = OperatorAction.from_hub('test/image-decode', (), {})
        op_info = {
            'operator': 'test/image-decode',
            'type': 'hub',
            'init_args': None,
            'init_kws': None,
            'tag': 'main'
        }
        self.assertEqual(op_action.serialize(), op_info)

    # pylint: disable=unnecessary-lambda-assignment
    def test_lambda(self):
        f = lambda x: x+1
        op_action = OperatorAction.from_lambda(f)
        op_info = {
            'operator': f,
            'type': 'lambda',
            'init_args': None,
            'init_kws': None,
            'tag': None
        }
        self.assertEqual(op_action.serialize(), op_info)

    def test_callable(self):
        def f():
            pass
        op_action = OperatorAction.from_callable(f)
        op_info = {
            'operator': f,
            'type': 'callable',
            'init_args': None,
            'init_kws': None,
            'tag': None
        }
        self.assertEqual(op_action.serialize(), op_info)
