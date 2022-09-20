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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
import unittest
from towhee.pipeline.operator_pool import OperatorPool

from towhee.pipeline.action import ops, Action


class TestOperatorContext(unittest.TestCase):
    """Test Action.
    """
    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._op_pool = OperatorPool(cache_path=cache_path)

    def tearDown(self):
        pass

    def test_lambda(self):
        lam = lambda x: x + 1
        s = Action.from_lambda(lam)
        self.assertEqual(s.type, 'lambda')
        res = s(1)
        self.assertEqual(res, 2)

    def test_callable(self):
        def cal(x):
            return x + 1
        s = Action.from_callable(cal)
        self.assertEqual(s.type, 'callable')
        res = s(1)
        self.assertEqual(res, 2)

    def test_hub(self):
        x = ops.local.add_operator(factor = 1)
        x = x.serialize()
        s = Action.from_hub(x['operator'], x['init_args'], x['init_kws'])
        self.assertEqual(s.type, 'hub')
        s.load_fn()
        res = s(1)
        self.assertEqual(res.sum, 2)

    def test_op_pool(self):
        x = ops.local.add_operator(factor = 1)
        x = x.serialize()
        s = Action.from_hub(x['operator'], x['init_args'], x['init_kws'])
        self.assertEqual(s.type, 'hub')
        s.load_from_op_pool(self._op_pool)
        res = s(1)
        self.assertEqual(res.sum, 2)
        s.release_to_pool()
        s.load_from_op_pool()
        res = s(1)
        self.assertEqual(res.sum, 2)
        s.release_to_pool()



if __name__ == '__main__':
    unittest.main()
