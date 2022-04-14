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

from towhee.hparam import param_scope
from towhee.hparam import HyperParameter


class StateMixin:
    """
    Mixin for state tracking.

    Examples:

    >>> from towhee import DataCollection, State
    >>> from towhee import param_scope
    >>> dc = DataCollection.range(10).set_state(State(a=1))
    >>> dc.get_state()
    {'a': 1}

    >>> dc = dc.map(lambda x: x+1).map(lambda x: x*2)
    >>> dc.get_state()
    {'a': 1}
    """

    def __init__(self):
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None:
            self.set_state(parent.get_state())

    def get_state(self):
        """
        Get the state storage for `DataCollection`

        Returns:
            State: the state storage
        """
        if hasattr(self, '_state') and isinstance(self._state, HyperParameter):
            return self._state
        return None

    def set_state(self, state):
        """
        Set the state storage for `DataCollection`

        Args:
            state (State): state storage

        Returns:
            DataCollection: data collection itself
        """
        self._state = state
        return self

    def set_training(self, state=None):
        """
        Set training mode for stateful operators

        Args:
            state (State, optional): Update the state storage. Defaults to None.

        Returns:
            DataCollection: data collection itself
        """
        if state is not None:
            self._state = state
        if self.get_state() is None:
            self._state = HyperParameter()
        self._state.__mode__ = 'training'
        return self

    def set_evaluating(self, state=None):
        """
        Set evaluating mode for stateful operators

        Args:
            state (State, optional): Update the state storage. Defaults to None.

        Returns:
            DataCollection: data collection itself
        """
        if state is not None:
            self._state = state
        if self.get_state() is None:
            self._state = HyperParameter()
        self._state.__mode__ = 'evaluating'
        return self

    def smap(self, op):
        self._dag[self._id] = ('smap',(op,), [])
        op.set_state(self._state)
        if self._state().__mode__('evaluating') == 'training':
            op.set_training(True)
            with param_scope() as hp:
                hp().towhee.data_collection.training = True
                for x in self:
                    op(x)
            op.set_training(False)
            op.fit()
        if hasattr(self._iterable, 'apply') and hasattr(op, '__dataframe_apply__'):
            return self._factory(op.__dataframe_apply__(self._iterable))

        return self._factory(map(op, self._iterable))


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
