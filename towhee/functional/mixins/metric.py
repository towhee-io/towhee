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


class Collector:
    """
    Collector class for metric
    """

    # pylint: disable=dangerous-default-value
    def __init__(self, metrics: list = [], labels: dict = {}, scores: dict = {}):
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
    for model in collector.scores:
        score_list = []
        for metric in collector.metrics:
            score_list.append(collector.scores[model][metric])
        scores_dict[model] = score_list
    return scores_dict


class MetricMixin:
    """
    Mixin for metric
    """

    # pylint: disable=import-outside-toplevel
    def __init__(self):
        self.collector = Collector()

    def with_metrics(self, metric_types: list = None):
        self.collector.metrics = metric_types
        return self

    def evaluate(self, actual: str, predicted: str, name: str):
        self.collector.add_labels({name: {'actual': actual, 'predicted': predicted}})
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
                re = sklearn_utils.recall_score(actual_list, predicted_list, average='weighted')
            elif metric_type == 'confusion_matrix':
                re = sklearn_utils.confusion_matrix(actual_list, predicted_list)
            score[name].update({metric_type: re})
        self.collector.add_scores(score)
        return self

    def report(self):
        import pandas as pd
        scores_dict = get_scores_dict(self.collector)
        df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=self.collector.metrics)
        df.style.highlight_max(color='lightgreen', axis=0)
        return self.collector.scores
