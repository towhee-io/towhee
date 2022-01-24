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
from typing import List

import torchmetrics as tm

__all__ = [
    'Metrics',
    'show_avaliable_metrics',
    'get_metric_by_name'
]

_metrics_impls = {}
_metrics_meta = {}

class Metrics:
    """
    `Metrics` is wrapper of popular metrics implementation used to evaluate the
    current prediction's performance by comparing it with the labels.

    Args:
        metric_name: (`str`):
            to indicate which metric is used to evalute.
    """

    @classmethod
    def _class_register(cls):
        Metrics._subclass_registry[cls.__repr__] = cls

    def __init__(self, metric_name: str):
        pass

    def update(self, pred, target) -> None:
        pass

    def compute(self) -> float:
        pass

    def reset(self) -> None:
        pass

class TMMetrics(Metrics):
    """
    `TMMetrics` use torchmetric as the implementation of metrics.

    Args:
        metric_name: (`str`):
            to indicate which metric is used to evalute.
    """
    @classmethod
    def get_tm_avaliable_metrics(cls):
        cls._metrics_list = {}
        cls._metrics_list['CatMetric'] = tm.CatMetric
        cls._metrics_list['MaxMetric'] = tm.MaxMetric
        cls._metrics_list['MeanMetric'] = tm.MeanMetric
        cls._metrics_list['MinMetric'] = tm.MinMetric
        cls._metrics_list['SumMetric'] = tm.SumMetric
        cls._metrics_list['PESQ'] = tm.PESQ
        cls._metrics_list['PIT'] = tm.PIT
        cls._metrics_list['SI_SDR'] = tm.SI_SDR
        cls._metrics_list['SI_SNR'] = tm.SI_SNR
        cls._metrics_list['SNR'] = tm.SNR
        cls._metrics_list['STOI'] = tm.STOI
        cls._metrics_list['AUC'] = tm.AUC
        cls._metrics_list['AUROC'] = tm.AUROC
        cls._metrics_list['F1'] = tm.F1
        cls._metrics_list['ROC'] = tm.ROC
        cls._metrics_list['Accuracy'] = tm.Accuracy
        cls._metrics_list['AveragePrecision'] = tm.AveragePrecision
        cls._metrics_list['BinnedAveragePrecision'] = tm.BinnedAveragePrecision
        cls._metrics_list['BinnedPrecisionRecallCurve'] = tm.BinnedPrecisionRecallCurve
        cls._metrics_list['BinnedRecallAtFixedPrecision'] = tm.BinnedRecallAtFixedPrecision
        cls._metrics_list['CalibrationError'] = tm.CalibrationError
        cls._metrics_list['CohenKappa'] = tm.CohenKappa
        cls._metrics_list['ConfusionMatrix'] = tm.ConfusionMatrix
        cls._metrics_list['FBeta'] = tm.FBeta
        cls._metrics_list['HammingDistance'] = tm.HammingDistance
        cls._metrics_list['Hinge'] = tm.Hinge
        cls._metrics_list['IoU'] = tm.IoU
        cls._metrics_list['KLDivergence'] = tm.KLDivergence
        cls._metrics_list['MatthewsCorrcoef'] = tm.MatthewsCorrcoef
        cls._metrics_list['Precision'] = tm.Precision
        cls._metrics_list['PrecisionRecallCurve'] = tm.PrecisionRecallCurve
        cls._metrics_list['Recall'] = tm.Recall
        cls._metrics_list['Specificity'] = tm.Specificity
        cls._metrics_list['StatScores'] = tm.StatScores
        cls._metrics_list['MetricCollection'] = tm.MetricCollection
        cls._metrics_list['MAP'] = tm.MAP
        cls._metrics_list['FID'] = tm.FID
        cls._metrics_list['IS'] = tm.IS
        cls._metrics_list['KID'] = tm.KID
        cls._metrics_list['LPIPS'] = tm.LPIPS
        cls._metrics_list['PSNR'] = tm.PSNR
        cls._metrics_list['SSIM'] = tm.SSIM
        cls._metrics_list['CosineSimilarity'] = tm.CosineSimilarity
        cls._metrics_list['ExplainedVariance'] = tm.ExplainedVariance
        cls._metrics_list['MeanAbsoluteError'] = tm.MeanAbsoluteError
        cls._metrics_list['MeanAbsolutePercentageError'] = tm.MeanAbsolutePercentageError
        cls._metrics_list['MeanSquaredError'] = tm.MeanSquaredError
        cls._metrics_list['MeanSquaredLogError'] = tm.MeanSquaredLogError
        cls._metrics_list['PearsonCorrcoef'] = tm.PearsonCorrcoef
        cls._metrics_list['R2Score'] = tm.R2Score
        cls._metrics_list['SpearmanCorrcoef'] = tm.SpearmanCorrcoef
        cls._metrics_list['SymmetricMeanAbsolutePercentageError'] = tm.SymmetricMeanAbsolutePercentageError
        cls._metrics_list['TweedieDevianceScore'] = tm.TweedieDevianceScore
        cls._metrics_list['RetrievalFallOut'] = tm.RetrievalFallOut
        cls._metrics_list['RetrievalHitRate'] = tm.RetrievalHitRate
        cls._metrics_list['RetrievalMAP'] = tm.RetrievalMAP
        cls._metrics_list['RetrievalMRR'] = tm.RetrievalMRR
        cls._metrics_list['RetrievalNormalizedDCG'] = tm.RetrievalNormalizedDCG
        cls._metrics_list['RetrievalPrecision'] = tm.RetrievalPrecision
        cls._metrics_list['RetrievalRecall'] = tm.RetrievalRecall
        cls._metrics_list['RetrievalRPrecision'] = tm.RetrievalRPrecision
        cls._metrics_list['WER'] = tm.WER
        cls._metrics_list['BERTScore'] = tm.BERTScore
        cls._metrics_list['BLEUScore'] = tm.BLEUScore
        cls._metrics_list['CharErrorRate'] = tm.CharErrorRate
        cls._metrics_list['ROUGEScore'] = tm.ROUGEScore
        cls._metrics_list['SacreBLEUScore'] = tm.SacreBLEUScore
        return list(cls._metrics_list.keys())

    def __init__(self, metric_name: str):
        super().__init__(metric_name)
        assert metric_name in TMMetrics._metrics_list
        self._metric = TMMetrics._metrics_list[metric_name]()

    def update(self, pred, target) -> None:
        self._metric.update(pred,target)

    def compute(self) -> str:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

_metrics_impls['TMMetrics'] = TMMetrics

def _generate_meta_info() -> None:
    global _metrics_meta
    res = {}
    res['TMMetrics'] = {}
    res['TMMetrics']['impl'] = 'pytorch'
    res['TMMetrics']['names'] = _metrics_impls['TMMetrics'].get_tm_avaliable_metrics()
    _metrics_meta = res

def show_avaliable_metrics() -> List:
    """
    Get the list of current avaliable metrics.
    """
    global _metrics_meta
    return _metrics_meta

def get_metric_by_name(metric_name: str, metric_impl: str = 'TMMetrics') -> Metrics:
    """
    Get the `Metrics` class by metric names, and metric_impl should be the specific implementation class.

    Args:
        metric_name: (`str`):
            the metric name used to evaluate. (e.g. Accuracy)
        metric_impl: (`str`)
            the metric implementation class name. (e.g. TMMetrics)
    """
    global _metrics_impls
    return _metrics_impls[metric_impl](metric_name)

_generate_meta_info()
