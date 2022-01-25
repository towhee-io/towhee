# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team and 2021 Zilliz.
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

import unittest

import torch

from towhee.trainer import metrics

class TestMetrics(unittest.TestCase):
    """
    TestMetrics
    """
    def test_Metrics(self) -> None:
        metrics.show_avaliable_metrics()
        metric = metrics.get_metric_by_name('Accuracy')

        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        metric.reset()
        metric.update(preds, target)
        metric.compute()
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main(verbosity=1)


