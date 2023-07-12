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
from concurrent.futures import ThreadPoolExecutor

from towhee import ops
from towhee.utils.hub_utils import HubUtils


class TestHubOp(unittest.TestCase):
    """
    Test HubOp import
    """

    def test_import(self):
        namespaces = [att for att in dir(ops) if not att.startswith('__')]

        # test __call__
        for n in namespaces:
            getattr(ops, n)()

        futures = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            for ns in namespaces:
                op_names = [op_name for op_name in dir(getattr(ops, ns)) if not op_name.startswith('__')]
                for name in op_names:
                    op = getattr(getattr(ops, ns), name)()
                    author, repo = op.name.split('/')
                    futures.append(pool.submit(HubUtils(author, repo).branch_tree, op.tag))

        for f in futures:
            self.assertIsNotNone(f.result())
