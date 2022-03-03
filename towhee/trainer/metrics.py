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
from typing import Any, Dict

import torchmetrics as tm
from torchmetrics.detection.map import MeanAveragePrecision

__all__ = [
    'Metrics',
    'TMMetrics',
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
        metric_name (`str`):
            Indicate which metric is used to evalute.
    """

    @classmethod
    def _class_register(cls):
        Metrics._subclass_registry[cls.__repr__] = cls

    def __init__(self, metric_name: str):
        pass

    def update(self, *_: Any, **__: Any) -> None:
        """
        Update the metric internal status.
        """
        raise NotImplementedError

    def compute(self) -> float:
        """
        Get the metric value.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the internal status of 'Metrics'.
        """
        raise NotImplementedError

    def to(self, *args, **kwargs):
        """
        Make 'Metrics' calculation on specific device.
        """
        raise NotImplementedError

class TMMetrics(Metrics):
    """
    `TMMetrics` use torchmetric as the implementation of metrics.

    Args:
        metric_name (`str`):
            Indicate which metric is used to evalute.

    Example:
        >>> from towhee.trainer.metrics import show_avaliable_metrics, get_metric_by_name
        >>> from towhee.trainer.metrics import TMMetrics
        >>> import torch
        >>> pred = torch.FloatTensor([1.0,1.0,1.0,0.0,0.0,0.0])
        >>> gt = torch.IntTensor([1,1,1,1,1,1])
        >>> metric =  TMMetrics('Accuracy')
        >>> metric.update(pred, gt)
        >>> pred = torch.FloatTensor([1.0,1.0,1.0,1.0,1.0,1.0])
        >>> gt = torch.IntTensor([1,1,1,1,1,1])
        >>> metric.update(pred, gt)
        >>> metric.compute().item()
        0.75
        >>> metric.reset()
        >>> pred = torch.FloatTensor([1.0,1.0,1.0,0.0,0.0,0.0])
        >>> gt = torch.IntTensor([1,1,1,1,1,1])
        >>> metric.update(pred, gt)
        >>> metric.compute().item()
        0.5
    """
    @classmethod
    def get_tm_avaliable_metrics(cls):
        cls._metrics_list = {}
        cls._metrics_list['CatMetric'] = tm.CatMetric
        cls._metrics_list['MaxMetric'] = tm.MaxMetric
        cls._metrics_list['MeanMetric'] = tm.MeanMetric
        cls._metrics_list['MinMetric'] = tm.MinMetric
        cls._metrics_list['SumMetric'] = tm.SumMetric
        cls._metrics_list['PIT'] = tm.PIT
        cls._metrics_list['SDR'] = tm.SDR
        cls._metrics_list['SI_SDR'] = tm.SI_SDR
        cls._metrics_list['SI_SNR'] = tm.SI_SNR
        cls._metrics_list['SNR'] = tm.SNR
        cls._metrics_list['PermutationInvariantTraining'] = tm.PermutationInvariantTraining
        cls._metrics_list['ScaleInvariantSignalDistortionRatio'] = tm.ScaleInvariantSignalDistortionRatio
        cls._metrics_list['ScaleInvariantSignalNoiseRatio'] = tm.ScaleInvariantSignalNoiseRatio
        cls._metrics_list['SignalDistortionRatio'] = tm.SignalDistortionRatio
        cls._metrics_list['SignalNoiseRatio'] = tm.SignalNoiseRatio
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
        cls._metrics_list['F1Score'] = tm.F1Score
        cls._metrics_list['FBeta'] = tm.FBeta
        cls._metrics_list['FBetaScore'] = tm.FBetaScore
        cls._metrics_list['HammingDistance'] = tm.HammingDistance
        cls._metrics_list['Hinge'] = tm.Hinge
        cls._metrics_list['HingeLoss'] = tm.HingeLoss
        cls._metrics_list['IoU'] = tm.IoU
        cls._metrics_list['JaccardIndex'] = tm.JaccardIndex
        cls._metrics_list['KLDivergence'] = tm.KLDivergence
        cls._metrics_list['MatthewsCorrcoef'] = tm.MatthewsCorrcoef
        cls._metrics_list['MatthewsCorrCoef'] = tm.MatthewsCorrCoef
        cls._metrics_list['Precision'] = tm.Precision
        cls._metrics_list['PrecisionRecallCurve'] = tm.PrecisionRecallCurve
        cls._metrics_list['Recall'] = tm.Recall
        cls._metrics_list['Specificity'] = tm.Specificity
        cls._metrics_list['PSNR'] = tm.PSNR
        cls._metrics_list['SSIM'] = tm.SSIM
        cls._metrics_list['MultiScaleStructuralSimilarityIndexMeasure'] = tm.MultiScaleStructuralSimilarityIndexMeasure
        cls._metrics_list['PeakSignalNoiseRatio'] = tm.PeakSignalNoiseRatio
        cls._metrics_list['StructuralSimilarityIndexMeasure'] = tm.StructuralSimilarityIndexMeasure
        cls._metrics_list['CosineSimilarity'] = tm.CosineSimilarity
        cls._metrics_list['ExplainedVariance'] = tm.ExplainedVariance
        cls._metrics_list['MeanAbsoluteError'] = tm.MeanAbsoluteError
        cls._metrics_list['MeanAbsolutePercentageError'] = tm.MeanAbsolutePercentageError
        cls._metrics_list['MeanSquaredError'] = tm.MeanSquaredError
        cls._metrics_list['MeanSquaredLogError'] = tm.MeanSquaredLogError
        cls._metrics_list['PearsonCorrcoef'] = tm.PearsonCorrcoef
        cls._metrics_list['PearsonCorrCoef'] = tm.PearsonCorrCoef
        cls._metrics_list['R2Score'] = tm.R2Score
        cls._metrics_list['SpearmanCorrcoef'] = tm.SpearmanCorrcoef
        cls._metrics_list['SpearmanCorrCoef'] = tm.SpearmanCorrCoef
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
        cls._metrics_list['BLEUScore'] = tm.BLEUScore
        cls._metrics_list['CharErrorRate'] = tm.CharErrorRate
        cls._metrics_list['CHRFScore'] = tm.CHRFScore
        cls._metrics_list['ExtendedEditDistance'] = tm.ExtendedEditDistance
        cls._metrics_list['MatchErrorRate'] = tm.MatchErrorRate
        cls._metrics_list['SacreBLEUScore'] = tm.SacreBLEUScore
        cls._metrics_list['SQuAD'] = tm.SQuAD
        cls._metrics_list['TranslationEditRate'] = tm.TranslationEditRate
        cls._metrics_list['WordErrorRate'] = tm.WordErrorRate
        cls._metrics_list['WordInfoLost'] = tm.WordInfoLost
        cls._metrics_list['WordInfoPreserved'] = tm.WordInfoPreserved
        cls._metrics_list['MinMaxMetric'] = tm.MinMaxMetric
        cls._metrics_list['MeanAveragePrecision'] = MeanAveragePrecision

        return list(cls._metrics_list.keys())

    def __init__(self, metric_name: str):
        super().__init__(metric_name)
        assert metric_name in TMMetrics._metrics_list
        self._metric = TMMetrics._metrics_list[metric_name]()

    def update(self, *_: Any, **__: Any) -> None:
        self._metric.update(*_,**__)

    def compute(self) -> str:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def to(self, *args, **kwargs):
        self._metric.to(*args, **kwargs)


_metrics_impls['TMMetrics'] = TMMetrics

def _generate_meta_info() -> None:
    global _metrics_meta
    res = {}
    res['TMMetrics'] = {}
    res['TMMetrics']['impl'] = 'pytorch'
    res['TMMetrics']['names'] = _metrics_impls['TMMetrics'].get_tm_avaliable_metrics()
    _metrics_meta = res

def show_avaliable_metrics() -> Dict:
    """
    Get the dict of current avaliable metrics.

    Returns:
        (`Dict`)
            A Dict which key is the class name of metrics implementation(e.g. TMMetrics),
            and the value is a dict has keys 'impl' and 'names'. 'impl' is the framework
            to implement this metric class and 'names' is the list of names of metrics
            which can be created to calculate the metric.

    Example:
        >>> from towhee.trainer.metrics import show_avaliable_metrics
        >>> aval_metrics = show_avaliable_metrics()
        >>> 'TMMetrics' in aval_metrics
        True
    """
    global _metrics_meta
    return _metrics_meta

def get_metric_by_name(metric_name: str, metric_impl: str = 'TMMetrics') -> Metrics:
    """
    Get the `Metrics` class by metric names, and metric_impl should be the specific implementation class.

    Args:
        metric_name (`str`):
            The metric name used to evaluate. (e.g. Accuracy)
        metric_impl (`str`):
            The metric implementation class name. (e.g. TMMetrics)

    Returns:
        (`towhee.trainer.metrics.Metrics`)
            Metrics instance which name matches the queried name.

    Example:
        >>> from towhee.trainer.metrics import show_avaliable_metrics, get_metric_by_name
        >>> from towhee.trainer.metrics import TMMetrics
        >>> metric = get_metric_by_name('Accuracy', 'TMMetrics')
        >>> isinstance(metric, TMMetrics)
        True
    """
    global _metrics_impls
    return _metrics_impls[metric_impl](metric_name)

_generate_meta_info()
