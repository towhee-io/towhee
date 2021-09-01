# coding=utf-8
# Copyright 2021 Zilliz and the HuggingFace Inc. team. All rights reserved.
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

import contextlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

#from debug_utils import DebugOption
from file_utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
    torch_required,
)
from vit_trainer_utils import EvaluationStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption
import logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Parameters:
        yaml_path (:obj:`str`):
            the yaml that configs the training arguments.
    """

    def __init__(self, yaml_path):
        # load yaml and parse the configurations
        raise NotImplementedError
    
    def __post_init__(self):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        raise NotImplementedError

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        raise NotImplementedError

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        raise NotImplementedError

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        raise NotImplementedError

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        raise NotImplementedError

    @property
    @torch_required
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs cores are available. One of:

        - :obj:`ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - :obj:`ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses :obj:`torch.nn.DataParallel`).
        - :obj:`ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          :obj:`torch.nn.DistributedDataParallel`).
        """
        raise NotImplementedError

    @property
    @torch_required
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        raise NotImplementedError

    @property
    @torch_required
    def process_index(self):
        """
        The index of the current process used.
        """
        raise NotImplementedError

    @property
    @torch_required
    def local_process_index(self):
        """
        The index of the local process used.
        """
        raise NotImplementedError

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        raise NotImplementedError

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        raise NotImplementedError

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to ``logging.INFO`` unless overridden by ``log_level`` argument.

        For the replica processes the log level defaults to ``logging.WARNING`` unless overridden by
        ``log_level_replica`` argument.

        The choice between the main and replica process settings is made according to the return value of
        ``should_log``.
        """

        raise NotImplementedError

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        raise NotImplementedError

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        raise NotImplementedError

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
            A context manager for torch distributed environment where on needs to do something on the main process,
            while blocking replicas, and when it's finished releasing the replicas.

            One such use is for ``datasets``'s ``map`` feature which to be efficient should be run once on the main
            process, which upon completion saves a cached version of results and which then automatically gets loaded
            by the replicas.

        Args:
            local (:obj:`bool`, `optional`, defaults to :obj:`True`):
                if :obj:`True` first means process of rank 0 of each node if :obj:`False` first means process of rank 0
                of node rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                ``local=False`` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (:obj:`str`, `optional`, defaults to ``"work"``):
                a work description to be used in debug logs

        """
        raise NotImplementedError

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        raise NotImplementedError

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        raise NotImplementedError

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        raise NotImplementedError

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        raise NotImplementedError


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
