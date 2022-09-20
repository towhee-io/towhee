from typing import Callable
from towhee.pipeline.action import Action
from towhee.runtime.runtime_pipeline import RuntimePipeline
from copy import deepcopy
import uuid
import json

# pylint: disable=protected-access, redefined-builtin, self-cls-assignment

class Pipeline:
    """Pipeline is a tool to create data transformation chains.

    Examples:
    >>> from towhee.pipeline.pipeline import Pipeline
    >>> from towhee.pipeline.action import ops
    >>> from pprint import pprint
    >>> def f(x):
    ...     return x + 1
    >>> p = ( Pipeline().input(('a', 'b'))
    ...     .map('a', 'c', ops.towhee.decode('a', b ='b'))
    ...     .flat_map(('a', 'b'), 'd', ops.towhee.clip('a', b ='b'))
    ...     .output('d')
    ... )
    >>> print(len(p._dag))
    4
    """
    def __init__(self):
        self._dag = {}
        self._prev = []
        self._prev_map = {}

    def __call__(self, *inputs):
        if getattr(self, 'r_pipe', None) is None:
            raise ValueError('Missing output node.')
        return self.r_pipe(*inputs)

    def set_config(self, config) -> 'Pipeline':
        """Set the config for a pipeline.

        Args:
            config (dict): The config to set.

        Returns:
            Pipeline: Pipeline with config set.
        """
        self._config = config

    @classmethod
    def input(cls, *schema) -> 'Pipeline':
        """Start a new pipeline chain.

        Args:
            schema (list): The schema for the values being inputted into the pipeline.

        Returns:
            Pipeline: Pipeline ready to be chained.
        """
        pipeline = cls()
        output_schema = tuple(schema)
        uid = '_input'
        pipeline._prev = [uid]
        pipeline._dag[uid] = {
            'inputs': output_schema,
            'outputs': output_schema,
            'iter_info': {
                'type': 'map',
                'param': None
            }
        }
        return pipeline

    def output(self, *input_schema, engine = 'local', workers = None) -> Callable:
        """Close and compile the pipeline chain.

        Args:
            input_schema (tuple): Which values to ouput.
            engine (str, optional): Which engine to run the pipeline, currently only 'local'. Defaults to 'local'.
            workers(int, optional): How many workers to use. Defaults to None.

        Returns:
            Callable: The rendered pipeline that can be called on inputs.
        """
        input_schema = tuple(input_schema)
        uid = '_output'
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev_map[uid] = None

        self._engine = engine
        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': input_schema,
            'iter_info': {
                'type': 'map',
                'param': None
            }
        }
        #TODO Return the DC once dc is decided.
        self._add_next_nodes()
        if engine == 'local':
            self.r_pipe = RuntimePipeline(self._dag, workers)
        return self


    def map(self, input_schema, output_schema, fn, config = None) -> 'Pipeline':
        """One to one map of function on inputs.

        Args:
            input_schema (tuple): The column/s to perform map on.
            output_schema (tuple): The column/s to output to.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the map. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
        """
        self = deepcopy(self)
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

        uid = uuid.uuid4().hex
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev = [uid]

        fn_action = self._to_action(fn)


        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': config
        }
        return self

    def flat_map(self, input_schema, output_schema, fn, config = None) -> 'Pipeline':
        """One to many map action.

        The opeartion might have a variable amount of outputs, each output is treated as
        a new row.

        Args:
            input_schema (tuple): The column/s to perform flat_map on.
            output_schema (tuple): The column/s to output to.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the flat_map. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
        """
        self = deepcopy(self)
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

        uid = uuid.uuid4().hex
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev = [uid]

        fn_action = self._to_action(fn)

        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'config': config
        }
        return self

    def filter(self, input_schema, output_schema, filter_columns, fn, config = None) -> 'Pipeline':
        """Filter the input columns based on the selected filter_columns and filter.

        Args:
            input_schema (tuple): The column/s to include in the filtering.
            output_schema (tuple): The output columns for post-filtering.
            filter_columns (str | list): Which columns from the input_schema to base filtering on.
            filter (Operation | lambda | callable): The filter.
            config (dict, optional): Config for the filter. Defaults to None.

        Returns:
            Pipeline: Pipeline with action added.
        """
        self = deepcopy(self)
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

        uid = uuid.uuid4().hex
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev = [uid]
        fn_action = self._to_action(fn)

        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': 'filter',
                'param': {'filter_columns': filter_columns}
            },
            'config': config
        }
        return self

    def time_window(self, input_schema, output_schema, timestamp_col, size, step, fn, config = None) -> 'Pipeline':
        """Perform action on time windows.

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
        self = deepcopy(self)
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

        uid = uuid.uuid4().hex
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev = [uid]

        fn_action = self._to_action(fn)

        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': 'time_window',
                'param': {'size': size, 'step': step, 'timestamp_col': timestamp_col}
            },
            'config': config
        }
        return self


    def window(self, input_schema, output_schema, size, step, fn, config = None) -> 'Pipeline':
        """Window execution of action.

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
        self = deepcopy(self)
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

        uid = uuid.uuid4().hex
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev = [uid]

        fn_action = self._to_action(fn)

        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': 'window',
                'param': {'size': size, 'step': step}
            },
            'config': config
        }
        return self

    def window_all(self, input_schema, output_schema, fn, config = None) -> 'Pipeline':
        """Read all rows as single window and perform action.

        Args:
            input_schema (tuple): The column/s to perform window_all map on.
            output_schema (tuple): The column/s to output to.
            fn (Operation | lambda | callable): The action to perform.
            config (dict, optional): Config for the window_all map. Defaults to None

        Returns:
            Pipeline: Pipeline with action added.
        """
        self = deepcopy(self)
        if isinstance(output_schema, str):
            output_schema = (output_schema,)
        if isinstance(input_schema, str):
            input_schema = (input_schema,)

        uid = uuid.uuid4().hex
        for x in self._prev:
            temp = self._prev_map.get(x, [])
            temp.append(uid)
            self._prev_map[x] = temp
        self._prev = [uid]

        fn_action = self._to_action(fn)

        self._dag[uid] = {
            'inputs': input_schema,
            'outputs': output_schema,
            'op_info': fn_action.serialize(),
            'iter_info': {
                'type': 'window_all',
                'param': None,
            },
            'config': config
        }
        return self

    def concat(self, *pipelines) -> 'Pipeline':
        """Read all rows as single window and perform action.

        Args:
            input_schema (tuple): The column/s to perform window_all map on.

        Returns:
            Pipeline: Pipeline with action added.
        """
        self = deepcopy(self)

        for pipe in pipelines:
            for key, value in pipe._dag.items():
                if key not in self._dag.keys():
                    self._dag[key] = value

            for key, value in pipe._prev_map.items():
                if key not in self._prev_map.keys():
                    self._prev_map[key] = pipe._prev_map[key]
                else:
                    self._prev_map[key].extend(pipe._prev_map[key])
            self._prev.extend(pipe._prev)
        return self

    @property
    def dag(self) -> 'Pipeline':
        return self._dag

    def _to_action(self, fn):
        if isinstance(fn, Action):
            return fn
        elif getattr(fn, '__name__', None) == '<lambda>':
            return Action.from_lambda(fn)
        elif callable(fn):
            return Action.from_callable(fn)

    def _add_next_nodes(self):
        for key, val in self._prev_map.items():
            self._dag[key]['next_nodes'] = val


    # # Different format for doing dag, keeping for possible future usage
    # def _add_next_nodes(self):
    #     def flatten(d):
    #         for i in d:
    #             yield from [i] if not isinstance(i, tuple) else flatten(i)
    #     mappings = {}
    #     for key, val in self._dag.items():
    #         if val['inputs'] is not None:
    #             inputs = list(flatten(val['inputs']))
    #             for y in inputs:
    #                 if y not in mappings:
    #                     mappings[y] = [key,]
    #                 else:
    #                     mappings[y].append(key)

    #     for key, val in self._dag.items():
    #         next_nodes = []
    #         if val['outputs'] is not None:
    #             for x in list(flatten(val['outputs'])):
    #                 if x in mappings:
    #                     next_nodes.extend(mappings[x])
    #         next_nodes = [x for x in next_nodes if x != key]
    #         next_nodes = [*set(next_nodes)]

    #         val['next_nodes'] = next_nodes if len(next_nodes) != 0 else None

    def _json(self):
        self._add_next_nodes()
        return json.dumps(self._dag, indent=4)
