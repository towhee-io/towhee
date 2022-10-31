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
from typing import List, Union
from copy import deepcopy

from towhee.runtime.operator_manager import OperatorAction
from towhee.runtime.factory import _OperatorWrapper
from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee.runtime.constants import (
    MapConst,
    WindowAllConst,
    WindowConst,
    FilterConst,
    TimeWindowConst,
    FlatMapConst,
    ConcatConst
)


# pylint: disable=protected-access
class Pipeline:
    """
    Pipeline is a tool to create data transformation chains.

    Args:
        dag (`dict`): The dag for the pipeline.
        clo_node (`str`): The close node in the pipeline dag, defaults to '_input'.
        config (`dict`): The config for the pipeline, defaults to None.
    """
    def __init__(self, dag, clo_node='_input', config=None):
        self._dag = dag
        self._clo_node = clo_node
        if config is None:
            self._engine = None
            self._parallel = None
            self._jit = None
            self._format_priority = None
        else:
            self.set_config(**config)

    @property
    def dag(self) -> dict:
        return self._dag

    @property
    def config(self) -> dict:
        return {
            'engine': self._engine,
            'parallel': self._parallel,
            'jit': self._jit,
            'format_priority': self._format_priority,
        }

    def clean_config(self):
        """
        Cleanup the config for a pipeline.
        """
        self._engine = None
        self._parallel = None
        self._jit = None
        self._format_priority = None

    def set_config(self,
                   engine: str = None,
                   parallel: int = None,
                   jit: Union[str, dict] = None,
                   format_priority: List[str] = None) -> 'Pipeline':
        """
        Set the config for a pipeline.

        Args:
            engine (`str`): The type of engine, defaults to None and run with local.
            parallel (int, optional): Set the number of parallel execution for the following calls, defaults to None.
            jit (Union[str, dict], optional): The parameter to just in time compile.
            format_priority (List[str], optional): The priority list of formats, defaults to None.

        Returns:
            Pipeline: Pipeline with config.

        Examples:
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = Pipeline.input('a').set_config(engine='local').map('a', 'b', lambda x: x+1)
            >>> pipe.config
            {'engine': 'local'}
        """
        self._engine = engine
        self._parallel = parallel
        self._jit = jit
        self._format_priority = format_priority
        return self

    @classmethod
    def input(cls, *schema) -> 'Pipeline':
        """
        Start a new pipeline chain.

        Args:
            schema (list): The schema for the values being inputted into the pipeline.

        Returns:
            Pipeline: Pipeline ready to be chained.

        Examples:
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = Pipeline.input('a', 'b', 'c')
        """
        dag_dict = {}
        output_schema = tuple(schema)
        uid = '_input'
        dag_dict[uid] = {
            'inputs': output_schema,
            'outputs': output_schema,
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': [],
        }
        return cls(dag_dict)

    # TODO: Run with the configuration.
    def output(self, *output_schema) -> 'RuntimePipeline':
        """
        Close and preload the pipeline, and ready to run with it.

        Args:
            output_schema (tuple): Which columns to output.

        Returns:
            RuntimePipeline: The runtime pipeline that can be called on inputs.

        Examples:
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = Pipeline.input('a').map('a', 'b', lambda x: x+1).output('b')
            >>> pipe(1).get()
            [2]
        """
        output_schema = tuple(output_schema)

        uid = '_output'
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': output_schema,
            'outputs': output_schema,
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None,
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)

        if self._engine == 'local' or self._engine is None:
            run_pipe = RuntimePipeline(dag_dict, self._parallel)
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
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = Pipeline.input('a').map('a', 'b', lambda x: x+1).output('a', 'b')
            >>> pipe(1).get()
            [1, 2]
        """
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

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
        return Pipeline(dag_dict, uid, self.config)

    def concat(self, *pipes: 'Pipeline') -> 'Pipeline':
        """
        Concat a pipeline to another pipeline/s.

        Args:
            pipes : one or more pipelines to concat.

        Returns:
            Pipeline: Pipeline to be concated.

        Examples:
            >>> pipe0 = Pipeline.input('a', 'b', 'c')
            >>> pipe1 = pipe0.map('a', 'd', lambda x: x+1)
            >>> pipe2 = pipe0.map(('b', 'c'), 'e', lambda x, y: x - y)
            >>> pipe3 = pipe2.concat(pipe1).output('d', 'e')
            >>> pipe3(1, 2, 3).get()
            [2, -1]
        """
        uid = uuid.uuid4().hex
        dag_dict = self._concat_dag(deepcopy(self._dag), pipes)
        dag_dict[uid] = {
            'inputs': (),
            'outputs': (),
            'iter_info': {
                'type': ConcatConst.name,
                'param': None
            },
            'next_nodes': [],
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)
        for pipe in pipes:
            dag_dict[pipe._clo_node]['next_nodes'].append(uid)

        config = self._concat_config(deepcopy(self.config), pipes)
        return Pipeline(dag_dict, uid, config)

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
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = (Pipeline.input('a')
            ...         .flat_map('a', 'b', lambda x: [y for y in x])
            ...         .output('b'))
            >>> res = pipe([1, 2, 3])
            >>> res.get()
            [1]
            >>> res.get()
            [2]
            >>> res.get()
            [3]
        """
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

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
        return Pipeline(dag_dict, uid, self.config)

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
            >>> from towhee.runtime.pipeline import Pipeline
            >>> def filter_func(num):
            ...     return num > 10
            >>> pipe = (Pipeline.input('a', 'c')
            ...         .filter('c', 'd', 'a', filter_func)
            ...         .output('d'))
            >>> pipe(1, 12).get()
            None
            >>> pipe(11, 12).get()
            [12]
        """
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

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
        return Pipeline(dag_dict, uid, self.config)

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
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = (Pipeline.input('n1', 'n2')
            ...         .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
            ...         .window(('n1', 'n2'), ('s1', 's2'), 2, 1, lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> res = pipe([1, 2, 3, 4], [2, 3, 4, 5])
            >>> res.get()
            [3, 5]
            >>> res.get()
            [5, 7]
        """
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

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
        return Pipeline(dag_dict, uid, self.config)

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
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = (Pipeline.input('n1', 'n2')
            ...         .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: [(a, b) for a, b in zip(x, y)])
            ...         .window_all(('n1', 'n2'), ('s1', 's2'), lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> pipe([1, 2, 3, 4], [2, 3, 4, 5]).get()
            [10, 14]
        """
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

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
        return Pipeline(dag_dict, uid, self.config)

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
            >>> from towhee.runtime.pipeline import Pipeline
            >>> pipe = (Pipeline.input('d')
            ...         .flat_map('d', ('n1', 'n2', 't'), lambda x: ((a, b, c) for a, b, c in x))
            ...         .time_window(('n1', 'n2'), ('s1', 's2'), 't', 3, 3, lambda x, y: (sum(x), sum(y)))
            ...         .output('s1', 's2'))
            >>> data = [(i, i+1, i * 1000) for i in range(11) if i < 3 or i > 7] #[(0, 1), (1, 2), (2, 3), (8, 9), (9, 10), (10, 11)]
            >>> res = pipe(data)
            >>> res.get() #[(0, 1), (1, 2), (2, 3)]
            [3, 6]
            >>> res.get() #(8, 9)
            [8, 9]
            >>> res.get() #(9, 10), (10, 11)
            [19, 21]
        """
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

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
        return Pipeline(dag_dict, uid, self.config)

    @staticmethod
    def _to_action(fn):
        if isinstance(fn, _OperatorWrapper):
            return OperatorAction.from_hub(fn.name, fn.init_args, fn.init_kws)
        elif getattr(fn, '__name__', None) == '<lambda>':
            return OperatorAction.from_lambda(fn)
        elif callable(fn):
            return OperatorAction.from_callable(fn)
        else:
            raise ValueError('Unknown operator, please make sure it is lambda, callable or operator with ops.')

    @staticmethod
    def _concat_dag(dag1, pipes):
        for pipe in pipes:
            dag2 = deepcopy(pipe.dag)
            same_nodes = dag1.keys() & dag2.keys()
            for name in same_nodes:
                dag2[name]['next_nodes'] += dag1[name]['next_nodes']
            dag1.update(dag2)
        return dag1

    @staticmethod
    def _concat_config(conf1, pipes):
        for pipe in pipes:
            conf2 = pipe.config
            for con_key in conf2.keys():
                if conf2[con_key] is None:
                    continue
                elif conf1[con_key] != conf2[con_key]:
                    raise ValueError('The config of each pipeline are inconsistent, please reset the config.')
                conf1.update(conf2)
        return conf1
