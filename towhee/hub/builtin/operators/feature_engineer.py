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

from towhee.engine import register
from towhee.operator.stateful_operator import StatefulOperator
# pylint: disable=import-outside-toplevel
# pylint: disable=useless-super-delegation
# pylint: disable=invalid-name


@register(name='builtin/standard_scaler')
class standard_scaler(StatefulOperator):
    """
    Standardize numerical features by removing the mean and scaling to unit variance.

    Examples:

    >>> from towhee import DataCollection, Entity
    >>> dc = (
    ...     DataCollection.range(10).map(lambda x: Entity(a=x))
    ...         .set_training()
    ...         .standard_scaler['a', 'b'](name='standard_scaler')
    ... )

    >>> [int(x.b*10) for x in dc.to_list()]
    [-15, -12, -8, -5, -1, 1, 5, 8, 12, 15]
    """
    def __init__(self, name: str = None):
        super().__init__(name)

    def fit(self):
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        data = np.array(self._data[0]).reshape([-1, 1])
        self._state.model = StandardScaler()
        self._state.model.fit(data)

    def predict(self, x):
        import numpy as np
        data = np.array([x]).reshape([-1, 1])
        return self._state.model.transform(data)


@register(name='builtin/num_discretizer')
class num_discretizer(StatefulOperator):
    """
    Bin numerical features into intervals.

    Examples:

    >>> from towhee import DataCollection, Entity
    >>> dc = (
    ...     DataCollection.range(10).map(lambda x: Entity(a=x))
    ...         .set_training()
    ...         .num_discretizer['a', 'b'](name='discretizer', n_bins=3)
    ... )

    >>> [x.b.nonzero()[1][0] for x in dc.to_list()]
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    """
    def __init__(self, name: str = None, n_bins=10, encode='onehot', strategy='quantile'):
        super().__init__(name)
        self._n_bins = n_bins
        self._encode = encode
        self._strategy = strategy

    def fit(self):
        from sklearn.preprocessing import KBinsDiscretizer
        import numpy as np
        data = np.array(self._data[0]).reshape([-1, 1])
        self._state.model = KBinsDiscretizer(n_bins=self._n_bins,
                                             encode=self._encode,
                                             strategy=self._strategy)
        self._state.model.fit(data)

    def predict(self, x):
        import numpy as np
        data = np.array([x]).reshape([-1, 1])
        return self._state.model.transform(data)


@register(name='builtin/cate_one_hot_encoder')
class cate_one_hot_encoder(StatefulOperator):
    """
    Standardize numerical features by removing the mean and scaling to unit variance.

    Examples:

    >>> from towhee import DataCollection, Entity
    >>> dc = (
    ...     DataCollection(['a','b','c','a','b']).map(lambda x: Entity(a=x))
    ...         .set_training()
    ...         .cate_one_hot_encoder['a', 'b'](name='one_hot_encoder')
    ... )

    >>> [x.b.nonzero()[1][0] for x in dc.to_list()]
    [0, 1, 2, 0, 1]
    """
    def __init__(self, name: str = None):
        super().__init__(name)

    def fit(self):
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np
        data = np.array(self._data[0]).reshape([-1, 1])
        self._state.model = OneHotEncoder()
        self._state.model.fit(data)

    def predict(self, x):
        import numpy as np
        data = np.array([x]).reshape([-1, 1])
        return self._state.model.transform(data)


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
