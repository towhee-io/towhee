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
# from towhee.operator import Operator


class StatefulOperator:
    """
    Stateful operator.

    Examples:

    >>> from towhee import register
    >>> from towhee import DataCollection, State
    >>> from towhee.functional.entity import Entity
    >>> import numpy as np
    >>> @register
    ... class my_normalize(StatefulOperator):
    ...     def __init__(self, name):
    ...         super().__init__(name=name)
    ...     def fit(self):
    ...         self._state._mu = np.mean(self._data[0])
    ...         self._state._std = np.std(self._data[0])
    ...     def predict(self, x):
    ...         return (x-self._state._mu())/self._state._std()

    >>> dc = (
    ...     DataCollection.range(10)
    ...         .set_training(State())
    ...         .map(lambda x: Entity(a=x))
    ...         .my_normalize['a', 'b'](name='mynorm')
    ... )

    >>> [int(x.b*10) for x in dc.to_list()]
    [-15, -12, -8, -5, -1, 1, 5, 8, 12, 15]
    >>> dc._state.mynorm._mu
    4.5
    """
    def __init__(self, name):
        super().__init__()
        self._name = name if name else self._get_default_name()
        self._data = None

    def _get_default_name(self):
        with param_scope() as hp:
            name = self.__class__.__name__.replace('_', '-')
            inputs = (hp.index[0] if isinstance(hp.index[0], str) else '-'.join(hp.index[0])).replace('_', '-')
            outputs = (hp.index[1] if isinstance(hp.index[1], str) else '-'.join(hp.index[1])).replace('_', '-')

            return '_'.join([name, inputs, outputs])

    def set_state(self, state):
        self._state = getattr(state, self._name)

    def feed(self, *arg):
        if self._data is None:
            self._data = [[x] for x in arg]
            return
        for i in range(len(arg)):
            self._data[i].append(arg[i])

    def fit(self, *arg):
        if len(arg) == 0 and self._data:
            return self.fit(*self._data)

    def predict(self, *arg):
        pass

    def __call__(self, *arg):
        with param_scope() as hp:
            training = hp().towhee.data_collection.training(False)
        if training:
            return self.feed(*arg)
        else:
            return self.predict(*arg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
