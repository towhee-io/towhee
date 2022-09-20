from towhee.pipeline.runtime_parser import runtime_parse
from towhee.engine.operator_loader import OperatorLoader
from towhee.pipeline.operator_pool import OperatorPool
# from os import path

# from towhee.engine.factory import from_op, op
# pylint: disable=protected-access

#TODO: Add different tag support.
tag = 'main'

class OperatorParser:
    """Class to allow runtime loading of operators.
    """
    def __init__(self):
        pass

    @classmethod
    def __getattr__(cls, name):

        @runtime_parse
        def wrapper(name, *args, **kwargs):
            return Action.from_hub(name, args, kwargs)

        return getattr(wrapper, name)

#TODO: make this towhee importable: from towhee import ops.
ops = OperatorParser()

class Action:
    """Action wrapper.

    Different types of operations are loaded into this wrapper for the DAG. Once the execution
    plan is decided, different operators can be loaded and run in different ways.
    """
    # TODO replace with _LazyOperatorLoader if deciding to use that.
    def __init__(self):
        self._loaded_fn = None
        self._type = None
        self._pool = None
        self._tag = 'main'

    @property
    def type(self):
        return self._type

    @staticmethod
    def from_hub(name, args, kwargs):
        """Create an Action for hub op.

        Args:
            name (str): The op name or the path to an op.
            args (list): The op args.
            kwargs (dict): The op kwargs.

        Returns:
            Action: The action.
        """
        action = Action()
        action._op_name = name
        action._op_hub_name = name.replace('.', '/').replace('_', '-')
        action._op_args = args
        action._op_kwargs = kwargs
        action._type = 'hub'
        return action

    #TODO: Deal with serialized input vs non_serialized
    @staticmethod
    def from_lambda(fn):
        """Create an Action for lambda op.

        Args:
            fn (lamda): The lambda function for op.

        Returns:
            Action: The action.
        """
        action = Action()
        action._fn = fn
        action._loaded_fn = fn
        action._type = 'lambda'
        return action

    #TODO: Deal with serialized input vs non_serialized
    @staticmethod
    def from_callable(fn):
        """Create an Action for callable op.

        Args:
            fn (callable): The callable function for op.

        Returns:
            Action: The action.
        """
        action = Action()
        action._fn = fn
        action._loaded_fn = fn
        action._type = 'callable'
        return action

    def serialize(self):
        if self._type == 'hub':
            return {
                'operator': self._op_hub_name,
                'type': self._type,
                'init_args': self._op_args if len(self._op_args) != 0 else None,
                'init_kws': self._op_kwargs if len(self._op_kwargs) != 0 else None,
                'tag': self._tag
            }
        elif self._type == 'lambda':
            raise ValueError('Lambda not supported yet.')
        elif self._type == 'callable':
            raise ValueError('Callable not supported yet.')

    def load_fn(self):
        if self._type == 'hub':
            loader = OperatorLoader()
            self._loaded_fn = loader.load_operator(self._op_hub_name, self._op_args, self._op_kwargs, tag = tag)

    def load_from_op_pool(self, pool: OperatorPool = None):
        self._pool = pool if pool is not None else self._pool
        if self._pool is None:
            raise ValueError('No pool to load from, need to pass pool in at least once.')
        self._loaded_fn = self._pool.acquire_op(self._op_hub_name, self._op_args, self._op_kwargs, tag = tag)

    def release_to_pool(self):
        self._pool.release_op(self._loaded_fn)
        self._loaded_fn = None

    def __call__(self, *args, **kwargs):
        if self._loaded_fn is None:
            self.load_fn()
        return self._loaded_fn(*args, **kwargs)

    def __str__(self):
        return str(list(self.__dict__.items()))
