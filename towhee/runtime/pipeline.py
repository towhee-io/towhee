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
        self._config = config

    @property
    def dag(self) -> dict:
        return self._dag

    @property
    def config(self) -> dict:
        return self._config

    def set_config(self, config) -> 'Pipeline':
        """
        Set the config for a pipeline.

        Args:
            config (dict): The config to set.

        Returns:
            Pipeline: Pipeline with config set.
        """
        self._config = config
        return self

    @classmethod
    def input(cls, *schema) -> 'Pipeline':
        """
        Start a new pipeline chain.

        Args:
            schema (list): The schema for the values being inputted into the pipeline.

        Returns:
            Pipeline: Pipeline ready to be chained.
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

    def output(self, *input_schema, engine='local', workers=None) -> 'RuntimePipeline':
        """
        Close and compile the pipeline chain.

        Args:
            input_schema (tuple): Which values to ouput.
            engine (str, optional): Which engine to run the pipeline, currently only 'local'. Defaults to 'local'.
            workers(int, optional): How many workers to use. Defaults to None.

        Returns:
            RuntimePipeline: The runtime pipeline that can be called on inputs.
        """
        input_schema = tuple(input_schema)
        if self._config is None:
            self._config = {'engine': engine}
        else:
            self._config['engine'] = engine

        uid = '_output'
        dag_dict = deepcopy(self._dag)
        dag_dict[uid] = {
            'inputs': input_schema,
            'outputs': input_schema,
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None,
        }
        dag_dict[self._clo_node]['next_nodes'].append(uid)

        if engine == 'local':
            run_pipe = RuntimePipeline(dag_dict, workers)
            run_pipe.preload()
        return run_pipe

    def concat(self, pipe: 'Pipeline') -> 'Pipeline':
        """
        concat a pipeline to an another.

        Args:
            pipe (tuple): The schema for the values being inputted into the pipeline.

        Returns:
            Pipeline: Pipeline ready to be chained.
        """
        # pylint: disable=protected-access
        uid = uuid.uuid4().hex
        dag_dict = self._concat_dag(deepcopy(self.dag), deepcopy(pipe._dag))
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
        dag_dict[pipe._clo_node]['next_nodes'].append(uid)

        config = self._concat_config(self._config, pipe._config)
        return Pipeline(dag_dict, uid, config)

    def map(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        One to one map of function on inputs.

        Args:
            input_schema (tuple): The column/s to perform map on.
            output_schema (tuple): The column/s to output to.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the map. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
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
        return Pipeline(dag_dict, uid, self._config)

    def flat_map(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        One to many map action.

        The operator might have a variable amount of outputs, each output is treated as a new row.

        Args:
            input_schema (tuple): The column/s to perform flat_map on.
            output_schema (tuple): The column/s to output to.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the flat_map. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
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
        return Pipeline(dag_dict, uid, self._config)

    def filter(self, input_schema, output_schema, filter_columns, fn, config=None) -> 'Pipeline':
        """
        Filter the input columns based on the selected filter_columns and filter.

        Args:
            input_schema (tuple): The column/s to include in the filtering.
            output_schema (tuple): The output columns for post-filtering.
            filter_columns (str | list): Which columns from the input_schema to base filtering on.
            fn (Operation | lambda | callable): The filter.
            config (dict, optional): Config for the filter. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
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
        return Pipeline(dag_dict, uid, self._config)

    def time_window(self, input_schema, output_schema, timestamp_col, size, step, fn, config=None) -> 'Pipeline':
        """
        Perform action on time windows.

        Args:
            input_schema (tuple): The column/s to include in the time_window.
            output_schema (tuple): The output columns for time_window action.
            timestamp_col (str): Which column to use for creating windows.
            size (int): size of window.
            step (int): how far to progress window.
            fn (Operation | lambda | callable): The function to perform on the window.
            config (dict, optional): Config for the time window. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
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
        return Pipeline(dag_dict, uid, self._config)

    def window(self, input_schema, output_schema, size, step, fn, config=None) -> 'Pipeline':
        """
        Window execution of action.

        Args:
            input_schema (tuple): The column/s to perform window map on.
            output_schema (tuple): The column/s to output to.
            size (int): How many rows per window
            step (int): How many rows to iterate after each window.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the window map. Defaults to None

        Returns:
            Pipeline: Pipeline with action added.
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
        return Pipeline(dag_dict, uid, self._config)

    def window_all(self, input_schema, output_schema, fn, config=None) -> 'Pipeline':
        """
        Read all rows as single window and perform action.

        Args:
            input_schema (tuple): The column/s to perform window_all map on.
            output_schema (tuple): The column/s to output to.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the window_all map. Defaults to None

        Returns:
            Pipeline: Pipeline with action added.
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
        return Pipeline(dag_dict, uid, self._config)

    @staticmethod
    def _to_action(fn):
        if isinstance(fn, _OperatorWrapper):
            return OperatorAction.from_hub(fn.name, fn.init_args, fn.init_kws)
        elif getattr(fn, '__name__', None) == '<lambda>':
            return OperatorAction.from_lambda(fn)
        elif callable(fn):
            return OperatorAction.from_callable(fn)

    @staticmethod
    def _concat_dag(dag1, dag2):
        same_nodes = dag1.keys() & dag2.keys()
        for name in same_nodes:
            dag2[name]['next_nodes'] += dag1[name]['next_nodes']
        dag1.update(dag2)
        return dag1

    @staticmethod
    def _concat_config(conf1, conf2):
        if conf1 is None:
            return conf2
        if conf2 is None:
            return conf1
        same_key = conf1.keys() & conf2.keys()
        for con_key in same_key:
            if conf1[con_key] != conf2[con_key]:
                raise ValueError('The config of each pipeline are inconsistent, please reset the config.')
        conf1.update(conf2)
        return conf1
