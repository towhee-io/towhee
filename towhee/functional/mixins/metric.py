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

from towhee.utils import sklearn_utils

class MetricMixin:
    """
    Mixin for metric
    """
    def with_accuracy(self, actual, predicted):
        return sklearn_utils.accuracy_score(actual, predicted)

    def with_recall(self, actual, predicted):
        return sklearn_utils.recall_score(actual, predicted, average='micro')

    def with_confusion_matrix(self, actual, predicted):
        return sklearn_utils.confusion_matrix(actual, predicted)

    def metric_summary(self, actual, predicted):
        metrics = {
            'accuracy': self.with_accuracy(actual, predicted),
            'recall': self.with_recall(actual, predicted),
            'confusion matrix': self.with_confusion_matrix(actual, predicted)
        }
        return  metrics
    