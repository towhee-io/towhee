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


import unittest
import copy
import threading

from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee.datacollection.data_collection import DataCollection


class TestNodesWithFlatMap(unittest.TestCase):
    """
    Test Nodes
    """

    dag = {
        '_input': {
            'inputs': ('scalar_num', 'nums'),
            'outputs': ('scalar_num', 'nums'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('nums', ),
            'outputs': ('num', ),
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'op_info': {
                'operator': lambda x: x,
                'type': 'lambda',
                'init_args': None,
                'init_kws': None,
                'tag': 'main'
            },
            'config': None,
            'next_nodes': ['op2']
        },
        'op2': {
            'inputs': ('scalar_num', 'num'),
            'outputs': ('sum', ),
            'iter_info': {
                'type': 'map',
                'param': None,
            },
            'op_info': {
                'operator': lambda x, y: x + y,
                'type': 'lambda',
                'init_args': None,
                'init_kws': None,
                'tag': 'main'
            },
            'config': {},
            'next_nodes': ['op3']
        },
        'op3': {
            'inputs': ('sum', ),
            'outputs': ('sum', ),
            'iter_info': {
                'type': 'filter',
                'param':
                    {'filter_by': ['sum']}
            },
            'op_info': {
                'operator': lambda x: x > 7,
                'type': 'lambda',
                'init_args': None,
                'init_kws': None,
                'tag': 'main'
            },
            'config': {},
            'next_nodes': ['_output']
        },
        '_output': {
            'inputs': ('sum', ),
            'outputs': ('sum', ),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None
        },
    }

    def test_simple(self):
        pipe = RuntimePipeline(self.dag)
        nums = [1, 2, 3, 4, 5]
        scalar_num = 4
        ret = pipe(scalar_num, nums)
        self.assertEqual(ret.size, 2)
        self.assertEqual(ret.schema, ['sum'])
        dc = DataCollection(ret)
        self.assertEqual(
            [item.sum for item in dc.to_list()],
            list(filter(lambda x: x > 7, list(map(lambda x: x + scalar_num, nums))))
        )

    def test_no_output(self):
        dag = copy.deepcopy(self.dag)
        dag['_output']['inputs'] = ()
        dag['_output']['outputs'] = ()
        pipe = RuntimePipeline(dag)
        nums = [1, 2, 3, 4, 5]
        scalar_num = 4
        ret = pipe(scalar_num, nums)
        self.assertEqual(ret.size, 0)

    def test_excpetion(self):
        pipe = RuntimePipeline(self.dag)
        with self.assertRaises(RuntimeError):
            pipe('x', [1])

    def test_multithread(self):
        pipe = RuntimePipeline(self.dag)
        pipe.preload()
        def run(pipe):
            nums = [1, 2, 3, 4, 5]
            scalar_num = 4
            ret = pipe(scalar_num, nums)
            self.assertEqual(ret.size, 2)
            self.assertEqual(ret.schema, ['sum'])
            dc = DataCollection(ret)
            self.assertEqual(
                [item.sum for item in dc.to_list()],
                list(filter(lambda x: x > 7, list(map(lambda x: x + scalar_num, nums))))
            )
        ts = []
        for _ in range(10):
            ts.append(threading.Thread(target=run, args=(pipe, )))
        for t in ts:
            t.start()
        for t in ts:
            t.join()
