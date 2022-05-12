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

import numpy as np

from towhee.engine import register
from towhee.operator.stateful_operator import StatefulOperator


# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name
@register(name='builtin/logistic_regression')
class logistic_regression(StatefulOperator):
    """
    Logistic Regression model encapsulate as an Operator.
    """
    def __init__(self, name: str = None, **kws):
        super().__init__(name=name)
        self._model_agrs = kws

    def fit(self):
        from towhee.utils.sklearn_utils import LogisticRegression
        from towhee.utils.scipy_utils import sparse
        X = sparse.vstack(self._data[0])
        y = np.array(self._data[1]).reshape([-1, 1])
        self._state.model = LogisticRegression(**self._model_agrs)
        self._state.model.fit(X, y)

    def predict(self, *arg):
        return self._state.model.predict(arg[0])


@register(name='builtin/random_forest')
class random_forest(StatefulOperator):
    """
    Random Forest Classifier model encapsulate as an Operator.
    """
    def __init__(self, name: str = None, **kws):
        super().__init__(name=name)
        self._model_agrs = kws

    def fit(self):
        from towhee.utils.sklearn_utils import RandomForestClassifier
        from towhee.utils.scipy_utils import sparse
        X = sparse.vstack(self._data[0])
        y = np.array(self._data[1]).reshape([-1, 1])
        self._state.model = RandomForestClassifier(**self._model_agrs)
        self._state.model.fit(X, y)

    def predict(self, *arg):
        return self._state.model.predict(arg[0])


@register(name='builtin/decision_tree')
class decision_tree(StatefulOperator):
    """
    Decision Tree Classifier model encapsulate as an Operator.
    """
    def __init__(self, name: str = None, **kws):
        super().__init__(name=name)
        self._model_agrs = kws

    def fit(self):
        from towhee.utils.sklearn_utils import DecisionTreeClassifier
        from towhee.utils.scipy_utils import sparse
        X = sparse.vstack(self._data[0])
        y = np.array(self._data[1]).reshape([-1, 1])
        self._state.model = DecisionTreeClassifier(**self._model_agrs)
        self._state.model.fit(X, y)

    def predict(self, *arg):
        return self._state.model.predict(arg[0])


@register(name='builtin/svc')
class svc(StatefulOperator):
    """
    SVM Classifier model encapsulate as an Operator.
    """
    def __init__(self, name: str = None, **kws):
        super().__init__(name=name)
        self._model_agrs = kws

    def fit(self):
        from towhee.utils.sklearn_utils import svm
        from towhee.utils.scipy_utils import sparse
        X = sparse.vstack(self._data[0])
        y = np.array(self._data[1]).reshape([-1, 1])
        self._state.model = svm.SVC(**self._model_agrs)
        self._state.model.fit(X, y)

    def predict(self, *arg):
        return self._state.model.predict(arg[0])
