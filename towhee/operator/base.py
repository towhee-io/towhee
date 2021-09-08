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


from abc import abstractmethod
from abc import ABC
from enum import Enum


# NotShareable:
#    Stateful operator

# Shareable:
#    Stateless operator
SharedType = Enum("SharedType", ("NotShareable", "Shareable"))


class OperatorBase(ABC):
    """
    Operator base class, implements __init__ and __call__,

    Examples:
        class AddOperator(OperatorBase):
            def __init__(self, factor: int):
                self._factor = factor

            def __call__(self, num) -> NamedTuple("Outputs", [("sum", int)]):
                Outputs = NamedTuple("Outputs", [("sum", int)])
                return Outputs(self._factor + num)
    """

    @abstractmethod
    def __init__(self):
        """
        Init operator, before a graph starts, the framework will call Operator __init__ function.

        Args:

        Returns:
            None

        Raises:
            An exception during __init__ can terminate the graph run.

        """
        pass

    @abstractmethod
    def __call__(self):
        """
        The framework calls __call__ function repeatedly for every input data.

        Args:


        Return:
            NamedTuple

        Raises:
            An exception during __init__ can terminate the graph run.
        """
        pass

    @property
    def shared_type(self):
        return SharedType.NotShared


class TorchNNOperatorBase(OperatorBase):
    """
    PyTorch model operator base
    """

    def train(self):
        """
        For training model
        """
        raise NotImplementedError
