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
from towhee.utils import sklearn_utils


@register(name='builtin/evaluate_metric')
class EvaluateMetric:
    """
    Convert a user-defined function as an operator and execute.

    Args:
        name (`str`):
            The model name.

    Examples:

    >>> from towhee.functional import DataCollection
    >>> from towhee.functional.entity import Entity
    >>> entities = [Entity(a=i, b=i) for i in range(5)]
    >>> dc = DataCollection(entities)
    >>> dc.with_metric(['recall', 'accuracy']).evaluate_metric[('a', 'b')](name='test')
    {'test': [0.0, 0.0]}
    """
    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kws):
        score = []
        for metric_type in self.collector.metric_types:
            if metric_type == 'accuracy':
                score.append(sklearn_utils.accuracy_score(args[0], args[1]))
            elif metric_type == 'recall':
                score.append(sklearn_utils.recall_score(args[0], args[1]))
            elif metric_type == 'confusion_matrix':
                score.append(sklearn_utils.confusion_matrix(args[0], args[1]))
        self.collector.score.update({self._name: score})
        return self


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
