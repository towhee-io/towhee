# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
Torch utilities for the Trainer class.
"""

import datetime
import json
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler

import logging


# this is used to suppress an undesired warning emitted by pytorch versions 1.4.2-1.7.0
try:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
except ImportError:
    SAVE_STATE_WARNING = ""

logger = logging.get_logger(__name__)


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    raise NotImplementedError


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    raise NotImplementedError


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    raise NotImplementedError


def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    raise NotImplementedError


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    raise NotImplementedError


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    raise NotImplementedError


def distributed_concat(tensor: "torch.Tensor", num_total_examples: Optional[int] = None) -> torch.Tensor:
    raise NotImplementedError


def distributed_broadcast_scalars(
    scalars: List[Union[int, float]], num_total_examples: Optional[int] = None
) -> torch.Tensor:
    raise NotImplementedError


def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the SAVE_STATE_WARNING
    raise NotImplementedError


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    raise NotImplementedError


class DistributedSamplerWithLoop(DistributedSampler):
    """
    Like a :obj:torch.utils.data.distributed.DistributedSampler` but loops at the end back to the beginning of the
    shuffled samples to make each process have a round multiple of batch_size samples.

    Args:
        dataset (:obj:`torch.utils.data.Dataset`):
            Dataset used for sampling.
        batch_size (:obj:`int`):
            The batch size used with this sampler
        kwargs:
            All other keyword arguments passed to :obj:`DistributedSampler`.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):
        warnings.warn(
            "SequentialDistributedSampler is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def nested_new_like(arrays, num_samples, padding_index=-100):
    """Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    raise NotImplementedError


def expand_like(arrays, new_seq_length, padding_index=-100):
    """Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    raise NotImplementedError


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    raise NotImplementedError


class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU by chunks.

    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on CPU at every
    step, our sampler will generate the following indices:

        :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then process 0, 1 and
    2 will be responsible of making predictions for the following samples:

        - P0: :obj:`[0, 1, 2, 3, 4, 5]`
        - P1: :obj:`[6, 7, 8, 9, 10, 11]`
        - P2: :obj:`[12, 13, 14, 15, 0, 1]`

    The first batch treated on each process will be

        - P0: :obj:`[0, 1]`
        - P1: :obj:`[6, 7]`
        - P2: :obj:`[12, 13]`

    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor) corresponding to
    the following indices:

        :obj:`[0, 1, 6, 7, 12, 13]`

    If we directly concatenate our results without taking any precautions, the user will then get the predictions for
    the indices in this order at the end of the prediction loop:

        :obj:`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    For some reason, that's not going to roll their boat. This class is there to solve that problem.

    Args:

        world_size (:obj:`int`):
            The number of processes used in the distributed training.
        num_samples (:obj:`int`):
            The number of samples in our dataset.
        make_multiple_of (:obj:`int`, `optional`):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
        padding_index (:obj:`int`, `optional`, defaults to -100):
            The padding index to use if the arrays don't all have the same sequence length.
    """

    def __init__(self, world_size, num_samples, make_multiple_of=None, padding_index=-100):
        raise NotImplementedError

    def add_arrays(self, arrays):
        """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
        raise NotImplementedError

    def _nested_set_tensors(self, storage, arrays):
        raise NotImplementedError

    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        raise NotImplementedError


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        raise NotImplementedError


def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None):
    """
    Return a list of indices so that each slice of :obj:`batch_size` consecutive indices correspond to elements of
    similar lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size :obj:`mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of :obj:`batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    raise NotImplementedError


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        raise NotImplementedError


class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """
    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        raise NotImplementedError


class ShardSampler(Sampler):
    """
    Sampler that shards batches between several processes. Dispatches indices batch by batch: on 2 processes with batch
    size 4, the first two batches are :obj:`[0, 1, 2, 3, 4, 5, 6, 7]` and :obj:`[8, 9, 10, 11, 12, 13, 14, 15]`, which
    shard into :obj:`[0, 1, 2, 3]` and :obj:`[8, 9, 10, 11]` for GPU-0 and :obj:`[4, 5, 6, 7]` and :obj:`[12, 13, 14,
    15]` for GPU-1.

    The sampler thus yields :obj:`[0, 1, 2, 3, 8, 9, 10, 11]` on GPU-0 and :obj:`[4, 5, 6, 7, 12, 13, 14, 15]` on
    GPU-1.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index

        self.total_batch_size = total_batch_size = batch_size * num_processes

        num_batches = len(dataset) // total_batch_size if drop_last else math.ceil(len(dataset) / total_batch_size)
        self.total_num_samples = num_batches * total_batch_size

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        # Each shard only sees a fraction of total_num_samples.
        raise NotImplementedError


class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch :obj:`IterableDataset` to generate samples for one of the processes only. Instances of this class
    will always yield a number of samples that is a round multiple of the actual batch size (which is :obj:`batch_size
    x num_processes`). Depending on the value of the :obj:`drop_last` attribute, it will either stop the iteration at
    the first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch
    size of 2:

    - the shard on process 0 will yield :obj:`[0, 1, 4, 5, 8, 9]` so will see batches :obj:`[0, 1]`, :obj:`[4, 5]`,
      :obj:`[8, 9]`
    - the shard on process 1 will yield :obj:`[2, 3, 6, 7, 10, 11]` so will see batches :obj:`[2, 3]`, :obj:`[6, 7]`,
      :obj:`[10, 11]`

    .. warning:

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a :obj:`torch.Generator` in a :obj:`generator` attribute of the
        :obj:`dataset` to generate your random numbers and call the
        :meth:`~towhee.trainer_pt_utils.IterableDatasetShard.set_epoch` method of this object. It will set the
        seed of this :obj:`generator` to :obj:`seed + epoch` on all processes before starting the iteration.
        Alternatively, you can also implement a :obj:`set_epoch()` method in your iterable dataset to deal with this.


    Args:
        dataset (:obj:`torch.utils.data.dataset.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The size of the batches per shard.
        drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.
        seed (:obj:`int`, `optional`, defaults to 0):
            A random seed that will be used for the random number generation in
            :meth:`~transformers.trainer_pt_utils.IterableDatasetShard.set_epoch`.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.seed = seed
        self.epoch = 0
        self.num_examples = 0

    def set_epoch(self, epoch):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


def _get_learning_rate(self):
    raise NotImplementedError


def _secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """
    raise NotImplementedError


def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (:obj:`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (:obj:`Dict[str, float]`): The reformatted metrics
    """
    raise NotImplementedError


def log_metrics(self, split, metrics):
    """
    Log metrics in a specially formatted way

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (:obj:`str`):
            Mode name: one of ``train``, ``eval``, ``test``
        metrics (:obj:`Dict[str, float]`):
            The metrics returned from train/evaluate/predictmetrics: metrics dict
    """
    raise NotImplementedError


def save_metrics(self, split, metrics, combined=True):
    """
    Save metrics into a json file for that split, e.g. ``train_results.json``.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (:obj:`str`):
            Mode/split name: one of ``train``, ``eval``, ``test``, ``all``
        metrics (:obj:`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Creates combined metrics by updating ``all_results.json`` with metrics of this call
    """
    raise NotImplementedError


def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

    Under distributed environment this is done only for a process with rank 0.
    """
    raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    raise NotImplementedError