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

    >>> from towhee.functional import DataCollection
    >>> from towhee.hparam import param_scope
    >>> dc = DataCollection.range(10).set_state(HyperParameter(a=1))
    >>> dc._state
    {'a': 1}

    >>> dc = dc.map(lambda x: x+1).map(lambda x: x*2)
    >>> dc._state
    {'a': 1}
    """

    def __init__(self):
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None:
            self.set_state(parent.get_state())

    def get_state(self):
        if hasattr(self, '_state') and isinstance(self._state, HyperParameter):
            return self._state
        return None

    def set_state(self, state):
        self._state = state
        return self

    def set_training(self, state=None):
        self._state = state if state is not None else HyperParameter()
        self._state.__mode__ = 'training'
        return self

    def set_evaluating(self, state):
        self._state = state if state is not None else HyperParameter()
        self._state.__mode__ = 'evaluating'
        return self

    def smap(self, op):
        op.set_state(self._state)
        if self._state().__mode__('evaluating') == 'training':
            with param_scope() as hp:
                hp().towhee.data_collection.training = True
                for x in self._iterable:
                    op(x)
            op.fit()
        return self.factory(map(op, self._iterable))


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
