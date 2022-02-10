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
import sys
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum

import torch
from torch import nn

from typing import List, Optional

from torch.nn.modules.loss import _Loss
from torch import optim
from torch.optim import Optimizer
import yaml

from towhee.trainer.callback import Callback
from towhee.trainer.utils import logging
from towhee.trainer.utils.trainer_utils import SchedulerType
from towhee.trainer import metrics

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
        default=None,
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
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    eval_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
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
    gradient_accumulation_num: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    prediction_accumulation_num: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    epoch_num: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )

    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means "
                    "that the data will be loaded in the main process."
        },
    )
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    lr: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_norm_grad: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    use_adafactor: bool = field(default=False, metadata={"help": "Wif Adafactor is used."})
    metric: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

    _n_gpu: int = field(init=False, repr=False, default=-1)
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})

    call_back_list: Optional[List[Callback]] = field(
        default=None, metadata={"help": "."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs "
                    "and batches to get to the same training data."
        },
    )
    loss: Optional[_Loss] = field(
        default=nn.CrossEntropyLoss(), metadata={"help": "pytorch loss"}
    )
    optimizer: Optional[Optimizer] = field(
        default=optim.Adam, metadata={"help": "pytorch optimizer"}
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    logging_nan_inf_filter: str = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    saving_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    saving_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    saving_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    saving_on_each_node: bool = field(
        default=False,
        metadata={
            "help": "When doing multi-node distributed training, whether to save models "
                    "and checkpoints on each node, or only on the main one"
        },
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

    def load_from_yaml(self, path2yaml: str = None):
        with open(path2yaml, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
            # args for training
            self.output_dir = conf["train"]["args"]["output_dir"]
            self.overwrite_output_dir = conf["train"]["args"]["overwrite_output_dir"]
            self.do_train = conf["train"]["args"]["do_train"]
            self.do_eval = conf["train"]["args"]["do_eval"]
            self.do_predict = conf["train"]["args"]["do_predict"]
            self.eval_strategy = conf["train"]["args"]["eval_strategy"]
            self.prediction_loss_only = conf["train"]["args"]["prediction_loss_only"]
            self.per_gpu_train_batch_size = conf["train"]["args"]["per_gpu_train_batch_size"]
            self.per_gpu_eval_batch_size = conf["train"]["args"]["per_gpu_eval_batch_size"]
            self.gradient_accumulation_num = conf["train"]["args"]["gradient_accumulation_num"]
            self.prediction_accumulation_num = conf["train"]["args"]["prediction_accumulation_num"]
            self.epoch_num = conf["train"]["args"]["epoch_num"]
            self.max_steps = conf["train"]["args"]["max_steps"]
            self.no_cuda = conf["train"]["args"]["no_cuda"]
            self.seed = conf["train"]["args"]["seed"]
            self.dataloader_drop_last = conf["train"]["args"]["dataloader_drop_last"]
            self.eval_steps = conf["train"]["args"]["eval_steps"]
            self.dataloader_num_workers = conf["train"]["args"]["dataloader_num_workers"]
            self.past_index = conf["train"]["args"]["past_index"]
            self.load_best_model_at_end = conf["train"]["args"]["load_best_model_at_end"]
            self.greater_is_better = conf["train"]["args"]["greater_is_better"]
            self.ignore_data_skip = conf["train"]["args"]["ignore_data_skip"]
            self.group_by_length = conf["train"]["args"]["group_by_length"]
            self.length_column_name = conf["train"]["args"]["length_column_name"]
            self.resume_from_checkpoint = conf["train"]["args"]["resume_from_checkpoint"]
            self.label_names = conf["train"]["args"]["label_names"]

            # args for learning
            s_temp = conf["learning"]["args"]["optimizer"]
            self.optimizer = getattr(sys.modules["torch.optim"],  s_temp.split(".")[-1])
            self.use_adafactor = conf["learning"]["args"]["use_adafactor"]
            self.lr_scheduler_type = conf["learning"]["args"]["lr_scheduler_type"]
            self.warmup_ratio = conf["learning"]["args"]["warmup_ratio"]
            self.warmup_steps = conf["learning"]["args"]["warmup_steps"]
            self.lr = conf["learning"]["args"]["lr"]
            self.weight_decay = conf["learning"]["args"]["weight_decay"]
            self.adam_beta1 = conf["learning"]["args"]["adam_beta1"]
            self.adam_beta2 = conf["learning"]["args"]["adam_beta2"]
            self.adam_epsilon = conf["learning"]["args"]["adam_epsilon"]
            self.max_norm_grad = conf["learning"]["args"]["max_norm_grad"]
            self.use_adafactor = conf["learning"]["args"]["use_adafactor"]

            # args for callbacks
            self.call_back_list = conf["callback"]["args"]["call_back_list"]

            # args for metrics
            if conf["metrics"]["args"]["metric"] == "None":
                self.metric = None
            else:
                self.metric = metrics.get_metric_by_name(conf["metrics"]["args"]["metric"])

            # args for logging
            self.logging_dir = conf["logging"]["args"]["logging_dir"]
            self.logging_strategy = conf["logging"]["args"]["logging_strategy"]
            self.logging_steps = conf["logging"]["args"]["logging_steps"]
            self.saving_strategy = conf["logging"]["args"]["saving_strategy"]
            self.saving_steps = conf["logging"]["args"]["saving_steps"]
            self.saving_total_limit = conf["logging"]["args"]["saving_total_limit"]
            self.saving_on_each_node = conf["logging"]["args"]["save_on_each_node"]
            self.disable_tqdm = conf["logging"]["args"]["disable_tqdm"]

    def save_to_yaml(self, path2yaml: str = None):
        train_args = {
            "args": {
                "output_dir": self.output_dir,
                "overwrite_output_dir": self.overwrite_output_dir,
                "do_train": self.do_train,
                "do_eval": self.do_eval,
                "do_predict": self.do_predict,
                "eval_strategy": self.eval_strategy,
                "prediction_loss_only": self.prediction_loss_only,
                "per_gpu_train_batch_size": self.per_gpu_train_batch_size,
                "per_gpu_eval_batch_size": self.per_gpu_eval_batch_size,
                "gradient_accumulation_num": self.gradient_accumulation_num,
                "prediction_accumulation_num": self.prediction_accumulation_num,
                "epoch_num": self.epoch_num,
                "max_steps": self.max_steps,
                "no_cuda": self.no_cuda,
                "seed": self.seed,
                "dataloader_drop_last": self.dataloader_drop_last,
                "eval_steps": self.eval_steps,
                "dataloader_num_workers": self.dataloader_num_workers,
                "past_index": self.past_index,
                "label_names": self.label_names,
                "load_best_model_at_end": self.load_best_model_at_end,
                "greater_is_better": self.greater_is_better,
                "ignore_data_skip": self.ignore_data_skip,
                "group_by_length": self.group_by_length,
                "length_column_name": self.length_column_name,
                "resume_from_checkpoint": self.resume_from_checkpoint,
            }
        }
        learning_args = {
            "args": {
                "optimizer": self.optimizer.__name__,
                "use_adafactor": self.use_adafactor,
                "lr_scheduler_type": self.lr_scheduler_type,
                "warmup_ratio": self.warmup_ratio,
                "warmup_steps": self.warmup_steps,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "adam_beta1": self.adam_beta1,
                "adam_beta2": self.adam_beta2,
                "adam_epsilon": self.adam_epsilon,
                "max_norm_grad": self.max_norm_grad,
            }
        }
        callback_args = {
            "args": {
                "call_back_list": self.call_back_list,
            }
        }
        metrics_args = {
            "args": {
                "metric": self.metric.__name__ if self.metric else "None",
            }
        }
        logging_args = {
            "args": {
                "logging_dir": self.logging_dir,
                "logging_strategy": self.logging_strategy,
                "logging_steps": self.logging_steps,
                "saving_strategy": self.saving_strategy,
                "saving_steps": self.saving_steps,
                "saving_total_limit": self.saving_total_limit,
                "save_on_each_node": self.saving_on_each_node,
                "disable_tqdm": self.disable_tqdm,
            }
        }
        conf = {
            "train": train_args,
            "learning": learning_args,
            "callback": callback_args,
            "metrics": metrics_args,
            "logging": logging_args,
        }
        with open(path2yaml, "w", encoding="utf-8") as file:
            yaml.dump(conf, file)
