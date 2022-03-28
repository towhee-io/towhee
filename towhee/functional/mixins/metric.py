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
    def __init__(self, metrics: list = None, labels: dict = {}, scores: dict = {}):
        self.metrics = metrics
        self.scores = scores
        self.labels = labels

    @metrics.setter
    def metrics(self, value: list):
        self.metrics = value

    @property
    def metrics(self):
        return self.metrics

    @property
    def scores(self):
        return self.scores

    @property
    def labels(self):
        return self.labels

    def add_scores(self, **kwargs):
        self.scores.update(kwargs)

    def add_labels(self, **kwargs):
        self.labels.update(kwargs)


def get_scores_dict(collector: Collector):
    scores_dict = {}
    for model in collector.scores:
        score = []
        for metric in collector.metrics:
            scores_dict[model] = score.append(collector.scores[model][metric])
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

    def report(self):
        import pandas as pd
        scores_dict = get_scores_dict(self.collector)
        df = pd.DataFrame.from_dict(scores_dict, orient='index', columns=self.collector.metrics)
        df.style.highlight_max(color='lightgreen', axis=0)
        return metrics_dict
