# Copyright 2020 The HuggingFace Team and 2021 Zilliz. All rights reserved.
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

import json
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from towhee.trainer.trainer_callback import TrainerCallback

import torch

from towhee.trainer.utils import logging

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class TrainingConfig:
    """
        Training arguments of CNN trainer.

        Parameters:
            output_dir (:obj:`str`):
                The output directory where the model predictions and checkpoints will be written.
            overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
                :obj:`output_dir` points to a checkpoint directory.
            prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
                When performing evaluation and generating predictions, only returns the loss.
            per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
                The batch size per GPU core/CPU for training.
            per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
                The batch size per GPU core/CPU for evaluation.
            eval_accumulation_steps (:obj:`int`, `optional`):
                Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
                left unset, the whole predictions are accumulated on GPU before being moved to the CPU (faster but
                requires more memory).
            num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
                the last epoch before stopping training).
            max_steps (:obj:`int`, `optional`, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides
                :obj:`num_train_epochs`.
            no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to not use CUDA even when it is available or not.
            eval_steps (:obj:`int`, `optional`):
                Number of update steps between two evaluations if :obj:`evaluation_strategy="steps"`.
            disable_tqdm (:obj:`bool`, `optional`):
                Whether or not to disable the tqdm progress bars.
            label_names (:obj:`List[str]`, `optional`):
                The list of keys in your dictionary of inputs that correspond to the labels.
        """


    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                    "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
                    "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    epoch_num: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    _n_gpu: int = field(init=False, repr=False, default=-1)
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})

    call_back_list: Optional[List[TrainerCallback]] = field(
        default=None, metadata={"help": "."}
    )


    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

        # if self.disable_tqdm is None:
        #     self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        self.should_save = True

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training.
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version."
            )
        per_device_batch_size = self.per_gpu_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._n_gpu = torch.cuda.device_count()

        return device

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices


    def load_from_yaml(self):
        pass

    def save_to_yaml(self):
        pass