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

import uuid
from copy import deepcopy

from towhee.runtime.check_utils import TupleForm
from towhee.runtime.operator_manager import OperatorAction
from towhee.runtime.factory import _OperatorWrapper
from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee.runtime.constants import (
    MapConst,
    WindowAllConst,
    WindowConst,
    ReduceConst,
    FilterConst,
    TimeWindowConst,
    FlatMapConst,
    ConcatConst,
    InputConst,
    OutputConst,
    OPName,
)


# pylint: disable=protected-access
class Pipeline:
    """
    Pipeline is a tool to create data transformation chains.

    Args:
        dag (`dict`): The dag for the pipeline.
        clo_node (`str`): The close node in the pipeline dag, defaults to '_input'.
    """
    def __init__(self, dag, clo_node=InputConst.name):
        self._dag = dag
        self._clo_node = clo_node

    @property
    def dag(self) -> dict:
        return self._dag

    @classmethod
    def input(cls, *schema) -> 'Pipeline':
        """
        Start a new pipeline chain.

        Args:
            schema (list): The schema for the values being inputted into the pipeline.

        Returns:
            Pipeline: Pipeline ready to be chained.

        Examples:
            >>> from towhee import pipe
            >>> p = pipe.input('a', 'b', 'c')
        """
        dag_dict = {}
        output_schema = cls._check_schema(schema)
        uid = InputConst.name
        dag_dict[uid] = cls._nop_node_dict(output_schema, output_schema)
        return cls(dag_dict)

    def output(self, *output_schema) -> 'RuntimePipeline':
        """
        Close and preload the pipeline, and ready to run with it.

        Args:
            output_schema (tuple): Which columns to output.
            config_kws (dict): The config for this pipeline.


        Returns:
            RuntimePipeline: The runtime pipeline that can be called on inputs.

        Examples:
            >>> from towhee import pipe
            >>> p = pipe.input('a').map('a', 'b', lambda x: x+1).output('b')
            >>> p(1).get()
            [2]
        """
        output_schema = self._check_schema(output_schema)

        uid = OutputConst.name
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = self._nop_node_dict(output_schema, output_schema)
        dag_dict[self._clo_node]['next_nodes'].append(uid)

        run_pipe = RuntimePipeline(dag_dict)
        run_pipe.preload()
        return run_pipe

    def map(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        One to one map of function on inputs.

        Args:
            input_schema (tuple): The input column/s of fn.
            output_schema (tuple): The output column/s of fn.
            fn (Operation | lambda | callable): The action to perform on the input_schema.
            config (dict, optional): Config for the map. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.

        Examples:
            >>> from towhee import pipe
            >>> p = pipe.input('a').map('a', 'b', lambda x: x+1).output('a', 'b')
            >>> p(1).get()
            [1, 2]
        """
        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': MapConst.name,
                'param': None
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    def concat(self, *pipes: 'Pipeline') -> 'Pipeline':
        """
        Concat a pipeline to another pipeline/s.

        Args:
            pipes : one or more pipelines to concat.

        Returns:
            Pipeline: Pipeline to be concated.

        Examples:
            >>> from towhee import pipe
            >>> p0 = pipe.input('a', 'b', 'c')
            >>> p1 = p0.map('a', 'd', lambda x: x+1)
            >>> p2 = p0.map(('b', 'c'), 'e', lambda x, y: x - y)
            >>> p3 = p2.concat(p1).output('d', 'e')
            >>> p3(1, 2, 3).get()
            [2, -1]
        """
        self._check_concat_pipe(pipes)
        uid = uuid.uuid4().hex
        dag_dict = self._concat_dag(deepcopy(self._dag), pipes)
        fn_action = self._to_action(ConcatConst.name)
        dag_dict[uid] = {
            'inputs': (),
            'outputs': (),
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': ConcatConst.name,
                'param': None,
            },
            'config': None,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        for pipe in pipes:
            dag_dict[pipe._clo_node]['next_nodes'].append(uid)

        return Pipeline(dag_dict, uid)

    def flat_map(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        One to many map action.

        The operator might have a variable amount of outputs, each output is treated as a new row.

        Args:
            input_schema (tuple): The input column/s of fn.
            output_schema (tuple): The output column/s of fn.
            fn (Operation | lambda | callable): The action to perform on the input_schema.
            config (dict, optional): Config for the flat_map. Defaults to None.

        Returns:
            Pipeline: Pipeline with flat_map action added.

        Examples:
            >>> from towhee import pipe
            >>> p = (pipe.input('a')
            ...         .flat_map('a', 'b', lambda x: [y for y in x])
            ...         .output('b'))
            >>> res = p([1, 2, 3])
            >>> res.get()
            [1]
            >>> res.get()
            [2]
            >>> res.get()
            [3]
        """
        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': FlatMapConst.name,
                'param': None
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    def filter(self, input_schema, output_schema, filter_columns, fn, config=None) -> 'Pipeline':
        """
        Filter the input columns based on the selected filter_columns.

        Args:
            input_schema (tuple): The input column/s before filter.
            output_schema (tuple): The output columns after filter, so the length of input_schema equals to the output_schema.
            filter_columns (str | tuple): Which columns to filter on.
            fn (Operation | lambda | callable): The action to perform on the filter_colums.
            config (dict, optional): Config for the filter. Defaults to None.

        Returns:
            Pipeline: Pipeline with filter action added.

        Examples:
            >>> from towhee import pipe
            >>> def filter_func(num):
            ...     return num > 10
            >>> p = (pipe.input('a', 'c')
            ...         .filter('c', 'd', 'a', filter_func)
            ...         .output('d'))
            >>> p(1, 12).get()
            None
            >>> p(11, 12).get()
            [12]
        """
        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)
        filter_columns = self._check_schema(filter_columns)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': FilterConst.name,
                'param': {FilterConst.param.filter_by: filter_columns}
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    def window(self, input_schema, output_schema, size, step, fn, config=None) -> 'Pipeline':
        """
        Window execution of action.

        Args:
            input_schema (tuple): The input column/s of fn.
            output_schema (tuple): The output column/s of fn.
            size (int): How many rows per window.
            step (int): How many rows to iterate after each window.
            fn (Operation | lambda | callable): The action to perform on the input_schema after window.
            config (dict, optional): Config for the window map. Defaults to None

        Returns:
            Pipeline: Pipeline with window action added.

        Examples:
            >>> from towhee import pipe
            >>> p = (pipe.input('n1', 'n2')
            ...         .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
            ...         .window(('n1', 'n2'), ('s1', 's2'), 2, 1, lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> res = p([1, 2, 3, 4], [2, 3, 4, 5])
            >>> res.get()
            [3, 5]
            >>> res.get()
            [5, 7]
        """
        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': WindowConst.name,
                'param': {WindowConst.param.size: size,
                          WindowConst.param.step: step}
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    def window_all(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        Read all rows as single window and perform action.

        Args:
            input_schema (tuple): The input column/s of fn.
            output_schema (tuple): The output column/s of fn.
            fn (Operation | lambda | callable): The action to perform on the input_schema after window all data.
            config (dict, optional): Config for the window_all. Defaults to None

        Returns:
            Pipeline: Pipeline with window_all action added.

        Examples:
            >>> from towhee import pipe
            >>> p = (pipe.input('n1', 'n2')
            ...         .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
            ...         .window_all(('n1', 'n2'), ('s1', 's2'), lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> p([1, 2, 3, 4], [2, 3, 4, 5]).get()
            [10, 14]
        """
        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': WindowAllConst.name,
                'param': None,
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    def reduce(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        Reduce the sequence to a single value.

        Args:
            input_schema (tuple): The input column/s of fn.
            output_schema (tuple): The output column/s of fn.
            fn (Operation | lambda | callable): The action to perform on the input_schema after window all data.
            config (dict, optional): Config for the window_all. Defaults to None

        Returns:
            Pipeline: Pipeline with reduce action added.

        Examples:
            >>> from towhee import pipe
            >>> p = (pipe.input('n1', 'n2')
            ...         .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
            ...         .reduce(('n1', 'n2'), ('s1', 's2'), lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> p([1, 2, 3, 4], [2, 3, 4, 5]).get()
            [10, 14]
        """
        if isinstance(fn, RuntimePipeline):
            raise RuntimeError("Reduce node doesn't support pipeline fn")

        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': ReduceConst.name,
                'param': None,
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    def time_window(self, input_schema, output_schema, timestamp_col, size, step, fn, config=None) -> 'Pipeline':
        """
        Perform action on time windows.

        Args:
            input_schema (tuple): The input column/s of fn.
            output_schema (tuple): The output columns to fn.
            timestamp_col (str): Which column to use for creating windows.
            size (int): size of window.
            step (int): how far to progress window.
            fn (Operation | lambda | callable): The action to perform on the input_schema
                                                after window the date with timestamp_col.
            config (dict, optional): Config for the time window. Defaults to None.

        Returns:
            Pipeline: Pipeline with time_window action added.

        Examples:
            >>> from towhee import pipe
            >>> p = (pipe.input('d')
            ...         .flat_map('d', ('n1', 'n2', 't'), lambda x: ((a, b, c) for a, b, c in x))
            ...         .time_window(('n1', 'n2'), ('s1', 's2'), 't', 3, 3, lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> data = [(i, i+1, i * 1000) for i in range(11) if i < 3 or i > 7] #[(0, 1), (1, 2), (2, 3), (8, 9), (9, 10), (10, 11)]
            >>> res = p(data)
            >>> res.get() #[(0, 1), (1, 2), (2, 3)]
            [3, 6]
            >>> res.get() #(8, 9)
            [8, 9]
            >>> res.get() #(9, 10), (10, 11)
            [19, 21]
        """
        output_schema = self._check_schema(output_schema)
        input_schema = self._check_schema(input_schema)

        uid = uuid.uuid4().hex
        fn_action = self._to_action(fn)
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': TimeWindowConst.name,
                'param': {TimeWindowConst.param.time_range_sec: size,
                          TimeWindowConst.param.time_step_sec: step,
                          TimeWindowConst.param.timestamp_col: timestamp_col}
            },
            'config': config,
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        return Pipeline(dag_dict, uid)

    @staticmethod
    def _to_action(fn):
        if fn in [OPName.NOP, ConcatConst.name]:
            return OperatorAction.from_builtin(fn)
        if isinstance(fn, _OperatorWrapper):
            return OperatorAction.from_hub(fn.name, fn.init_args, fn.init_kws, fn.tag, fn.is_latest)
        if isinstance(fn, RuntimePipeline):
            return OperatorAction.from_pipeline(fn)
        if getattr(fn, '__name__', None) == '<lambda>':
            return OperatorAction.from_lambda(fn)
        if callable(fn):
            return OperatorAction.from_callable(fn)
        raise ValueError('Unknown operator, please make sure it is lambda, callable or operator with ops.')

    @staticmethod
    def _nop_node_dict(input_schema, output_schema):
        fn_action = Pipeline._to_action(OPName.NOP)
        node_dict = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': MapConst.name,
                'param': None,
            },
            'config': None,
            'next_nodes': [],
        }
        return node_dict

    @staticmethod
    def _check_concat_pipe(pipes):
        if len(pipes) == 0:
            raise ValueError('The parameter of concat cannot be None.')
        for pipe in pipes:
            if not isinstance(pipe, Pipeline):
                raise ValueError(f'{pipe} is invalid, the parameter of concat must be Pipeline.')

    @staticmethod
    def _concat_dag(dag1, pipes):
        for pipe in pipes:
            dag2 = deepcopy(pipe.dag)
            same_nodes = dag1.keys() & dag2.keys()
            for name in same_nodes:
                dag2[name]['next_nodes'] = list(set(dag2[name]['next_nodes'] + dag1[name]['next_nodes']))
            dag1.update(dag2)
        return dag1

    @staticmethod
    def _check_schema(schema):
        return TupleForm(schema_data=schema).schema_data
