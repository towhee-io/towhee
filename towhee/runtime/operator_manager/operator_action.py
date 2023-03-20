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

from towhee.runtime.dag_repr import DAGRepr
from towhee.runtime.constants import OPType

# pylint: disable=protected-access
class OperatorAction:
    """
    Action wrapper.

    Different types of operations are loaded into this wrapper for the DAG, each
    operators can be loaded and run in different ways.
    """
    def __init__(self):
        self._loaded_fn = None
        self._type = None
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
        action = OperatorAction()
        action._op_name = name
        action._op_args = args
        action._op_kwargs = kwargs
        action._type = 'hub'
        return action

    @staticmethod
    def from_builtin(name, args=None, kwargs=None):
        """Create an Action for hub op.

        Args:
            name (str): The op name or the path to an op.
            args (list): The op args.
            kwargs (dict): The op kwargs.

        Returns:
            Action: The action.
        """
        action = OperatorAction()
        action._op_name = name
        action._op_args = args
        action._op_kwargs = kwargs
        action._type = OPType.BUILTIN
        action._tag = None
        return action

    @staticmethod
    def from_pipeline(fn):
        """Create an Action for callable op.

        Args:
            fn (callable): The callable function for op.

        Returns:
            Action: The action.
        """
        action = OperatorAction()
        action._dag_dict = fn.dag_repr.dag_dict
        action._top_sort = DAGRepr.get_top_sort(fn.dag_repr.nodes)
        action._type = OPType.PIPELINE
        return action

    # TODO: Deal with serialized input vs non_serialized
    @staticmethod
    def from_lambda(fn):
        """Create an Action for lambda op.

        Args:
            fn (lambda): The lambda function for op.

        Returns:
            Action: The action.
        """
        action = OperatorAction()
        action._fn = fn
        action._loaded_fn = fn
        action._type = OPType.LAMBDA
        return action

    # TODO: Deal with serialized input vs non_serialized
    @staticmethod
    def from_callable(fn):
        """Create an Action for callable op.

        Args:
            fn (callable): The callable function for op.

        Returns:
            Action: The action.
        """
        action = OperatorAction()
        action._fn = fn
        action._loaded_fn = fn
        action._type = OPType.CALLABLE
        return action

    def serialize(self):
        if self._type in [OPType.HUB, OPType.BUILTIN]:
            return {
                'operator': self._op_name,
                'type': self._type,
                'init_args': self._op_args if self._op_args and len(self._op_args) != 0 else None,
                'init_kws': self._op_kwargs if self._op_kwargs and len(self._op_kwargs) != 0 else None,
                'tag': self._tag
            }
        elif self._type in [OPType.LAMBDA, OPType.CALLABLE]:
            return {
                'operator': self._fn,
                'type': self._type,
                'init_args': None,
                'init_kws': None,
                'tag': None
            }
        elif self._type == OPType.PIPELINE:
            return {
                'dag': self._dag_dict,
                'top_sort': self._top_sort,
                'type': self._type,
            }
        else:
            raise KeyError(f'Operator type {self._type} is not valid.')
