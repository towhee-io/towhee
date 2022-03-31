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
import pandas as pd
from typing import Tuple
from IPython.display import display

from towhee.hparam import param_scope


class Collector:
    """
    Collector class for metric
    """

    # pylint: disable=dangerous-default-value
    def __init__(self,
                 metrics: list = [],
                 labels: dict = {},
                 scores: dict = {}):
        self.metrics = metrics
        self.scores = scores
        self.labels = labels

    def add_metrics(self, value: str):
        self.metrics.append(value)

    def add_scores(self, value: dict):
        self.scores.update(value)

    def add_labels(self, value: dict):
        self.labels.update(value)


def get_scores_dict(collector: Collector):
    scores_dict = {}
    matrix = {}
    for model in collector.scores:
        score_list = []
        for metric in collector.metrics:
            if metric == 'confusion_matrix':
                matrix[model] = collector.scores[model][metric]
            else:
                score_list.append(collector.scores[model][metric])
        scores_dict[model] = score_list

    return matrix, scores_dict


def _evaluate_callback(self):

    def wrapper(_: str, index: Tuple[str], *arg, **kws):
        # pylint: disable=import-outside-toplevel
        # pylint: disable=unused-argument
        actual, predicted = index
        name = None
        if 'name' in kws:
            name = kws['name']
        elif arg:
            name, = arg
        self.collector.add_labels(
            {name: {
                'actual': actual,
                'predicted': predicted
            }})
        score = {name: {}}
        actual_list = []
        predicted_list = []
        for x in self:
            actual_list.append(getattr(x, actual))
            predicted_list.append(getattr(x, predicted))

        from towhee.utils import sklearn_utils
        for metric_type in self.collector.metrics:
            if metric_type == 'accuracy':
                re = sklearn_utils.accuracy_score(actual_list, predicted_list)
            elif metric_type == 'recall':
                re = sklearn_utils.recall_score(actual_list,
                                                predicted_list,
                                                average='weighted')
            elif metric_type == 'confusion_matrix':
                re = sklearn_utils.confusion_matrix(actual_list,
                                                    predicted_list)
            score[name].update({metric_type: re})
        self.collector.add_scores(score)
        return self

    return wrapper


class MetricMixin:
    """
    Mixin for metric
    """

    # pylint: disable=import-outside-toplevel
    def __init__(self):
        self.collector = Collector()
        self.evaluate = param_scope().callholder(_evaluate_callback(self))

    def with_metrics(self, metric_types: list = None):
        self.collector.metrics = metric_types
        return self

    def report(self):
        """
        report the metric scores

        Examples:

        >>> from towhee import DataCollection
        >>> from towhee import Entity
        >>> dc = DataCollection([Entity(a=a, b=b, c=c) for a, b, c in zip([0,1,1,0,0], [0,1,1,1,0], [0,1,1,0,0])])
        >>> dc.with_metrics(['accuracy', 'recall']).evaluate['a', 'c'](name='lr').evaluate['a', 'b'](name='rf').report()
        <pandas.io...>
        {'lr': {'accuracy': 1.0, 'recall': 1.0}, 'rf': {'accuracy': 0.8, 'recall': 0.8}}
        """
        from towhee.utils import sklearn_utils
        matrix_dict, scores_dict = get_scores_dict(self.collector)
        if matrix_dict:
            self.collector.metrics.remove('confusion_matrix')
            for model in matrix_dict.items():
                cm = matrix_dict[model].astype('float') / matrix_dict[model].sum(axis=1)[:, np.newaxis] #normalize
                disp = sklearn_utils.ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                disp.ax_.set_title(f'Confusion matrix for Model :{model}')
        if scores_dict: # doctest:+ELLIPSIS
            df = pd.DataFrame.from_dict(scores_dict,
                                        orient='index',
                                        columns=self.collector.metrics)
            df = df.style.highlight_max(color='lightgreen', axis=0)
            display(df)
        return self.collector.scores


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
