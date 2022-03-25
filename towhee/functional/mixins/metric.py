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
    def __init__(self, **kwargs):
        self.collector = kwargs

    def __getattr__(self, name):
        return self.collector[name]

    def add(self, **kwargs):
        self.collector.update(kwargs)


class MetricMixin:
    """
    Mixin for metric
    """

    # pylint: disable=import-outside-toplevel
    def __init__(self):
        self.collector = Collector(score={})

    def with_metrics(self, metric_types: list = None):
        self.collector.add(metric_types=metric_types)
        return self

    def report(self):
        import pandas as pd
        metrics_dict = self.collector.score
        df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=self.collector.metric_types)
        df.style.highlight_max(color='lightgreen', axis=0)
        return metrics_dict
