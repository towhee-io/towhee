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

import json
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum

import torch

from typing import Optional, Dict, Any, Union
import yaml

from towhee.utils.log import trainer_log

HELP = "help"
CATEGORY = "category"
OTHER_CATEGORY = "other"

def _get_attr_str(obj, attr_name):
    fld = getattr(obj, attr_name)
    if isinstance(fld, Enum):
        return fld.value
    else:
        return fld


def dump_default_yaml(yaml_path):
    """
    dump a default yaml, which can be overridden by the custom operator.
    """
    training_config = TrainingConfig()
    training_config.save_to_yaml(path2yaml=yaml_path)
    trainer_log.info("dump default yaml to %s", yaml_path)


def _early_stopping_factory():
    return {
        "monitor": "eval_epoch_metric",
        "patience": 4,
        "mode": "max"
    }


def _tensorboard_factory():
    return {
        "log_dir": None,  # if None, the tensorboard will make a dir `./runs` as the log_dir
        "comment": ""
    }


def _model_checkpoint_factory():
    return {
        "every_n_epoch": 1
    }


@dataclass
class TrainingConfig:
    """
    Training configuration, it can be defined in a yaml file

    :param output_dir: the output directory
    :type output_dir: str

    :param overwrite_output_dir: whether overwrite output directory or not
    :type overwrite_output_dir: bool

    :param eval_strategy: the evaluation strategy
    :type eval_strategy: str

    :param eval_steps: run an evaluation every X steps
    :type eval_steps: int

    :param batch_size: the train_dataloader of trainer
    :type batch_size:  int

    :param val_batch_size: batch size for evaluation
    :type val_batch_size:  int

    :param seed: random seed that will be set at the beginning of training
    :type seed:  int

    :param epoch_num: total number of training epochs to perform
    :type epoch_num: float

    :param max_steps: If > 0: set total number of training steps to perform. Override num_train_epochs
    :type max_steps: int

    :param dataloader_pin_memory: Whether or not to pin memory for DataLoader
    :type dataloader_pin_memory: bool

    :param dataloader_drop_last: Drop the last incomplete batch if it is not divisible by the batch size
    :type dataloader_drop_last: bool

    :param dataloader_num_workers: Number of subprocesses to use for data loading
    :type dataloader_num_workers: int

    :param resume_from_checkpoint: The path to a folder with a valid checkpoint for your model
    :type resume_from_checkpoint: str

    :param lr: the initial learning rate for AdamW
    :type lr: float

    :param metric: the metric to use to compare two different models
    :type metric: str

    :param print_steps: if None, use the tqdm progress bar, otherwise it will
    print the logs on the screen every `print_steps`
    :type print_steps: int

    :param load_best_model_at_end: Whether or not to load the best model found during training at the end of training
    :type load_best_model_at_end: bool

    :param early_stopping: early stopping
    :type early_stopping: Union[dict, str]

    :param model_checkpoint: model checkpoint
    :type model_checkpoint: Union[dict, str]

    :param tensorboard: tensorboard
    :type tensorboard: Union[dict, str]

    :param loss: pytorch loss in torch.nn package
    :type loss: Union[str, Dict[str, Any]]

    :param optimizer: pytorch optimizer Class name in torch.optim package
    :type optimizer: Union[str, Dict[str, Any]]

    :param lr_scheduler_type: the scheduler type to be used
    :type lr_scheduler_type: str

    :param warmup_ratio: linear warmup over warmup_ratio fraction of total steps
    :type warmup_ratio: float

    :param warmup_steps: linear warmup over warmup_step
    :type warmup_steps: int

    :param logging_dir: the tensorboard log directory
    :type logging_dir: str

    :param logging_strategy: the logging strategy to be used
    :type logging_strategy: str

    :param device_str: the cpu or cuda device to be used
    :type device_str: str

    :param n_gpu: should be specified when device_str is
    :type n_gpu: int

    :param sync_bn: will be work if device_str is `cuda`, the True sync_bn would make training slower but acc better
    :type sync_bn: bool

    :example:
        >>> from towhee.trainer.training_config import TrainingConfig
        >>> training_args = TrainingConfig(
        ...     output_dir='./ResNet50',
        ...     overwrite_output_dir=True,
        ...     epoch_num=1,
        ...     batch_size=4,
        ...     dataloader_num_workers=0
        ... )
    """
    output_dir: str = field(
        default="./output_dir",
        metadata={HELP: "The output directory where the model predictions and checkpoints will be written.",
                  CATEGORY: "train"},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            HELP: (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            ),
            CATEGORY: "train"
        },
    )
    eval_strategy: str = field(
        default="epoch",
        metadata={HELP: "The evaluation strategy. It can be `steps`, `epoch` or `no`,", CATEGORY: "train"},
    )
    eval_steps: int = field(default=None, metadata={HELP: "Run an evaluation every X steps.", CATEGORY: "train"})
    batch_size: Optional[int] = field(
        default=8,
        metadata={
            HELP: "batch size for training.",
            CATEGORY: "train"
        }
    )
    val_batch_size: Optional[int] = field(
        default=-1,
        metadata={
            HELP: "batch size for eval.",
            CATEGORY: "train"
        }
    )
    seed: int = field(default=42,
                      metadata={HELP: "Random seed that will be set at the beginning of training.", CATEGORY: "train"})

    epoch_num: float = field(default=2,
                             metadata={HELP: "Total number of training epochs to perform.", CATEGORY: "train"})
    max_steps: int = field(
        default=-1,
        metadata={HELP: "If > 0: set total number of training steps to perform. Override num_train_epochs.",
                  CATEGORY: "train"},
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={HELP: "Whether or not to pin memory for DataLoader.", CATEGORY: "train"}
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={HELP: "Drop the last incomplete batch if it is not divisible by the batch size.", CATEGORY: "train"}
    )
    dataloader_num_workers: int = field(
        default=-1,
        metadata={
            HELP: "Number of subprocesses to use for data loading."
                  "default -1 means using all the cpu kernels,"
                  "it will greatly improve the speed when distributed training."
                  "0 means that the data will be loaded in the main process.",
            CATEGORY: "train"
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={HELP: "The path to a folder with a valid checkpoint for your model.", CATEGORY: "train"},
    )
    lr: float = field(default=5e-5, metadata={HELP: "The initial learning rate for AdamW.", CATEGORY: "learning"})
    metric: Optional[str] = field(
        default="Accuracy", metadata={HELP: "The metric to use to compare two different models.", CATEGORY: "metrics"}
    )

    print_steps: Optional[int] = field(
        default=None, metadata={
            HELP: "if None, use the tqdm progress bar, otherwise it will print the logs on the screen every `print_steps`",
            CATEGORY: "logging"}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={HELP: "Whether or not to load the best model found during training at the end of training.",
                  CATEGORY: "train"},
    )
    early_stopping: Union[dict, str] = field(
        default_factory=_early_stopping_factory, metadata={HELP: ".", CATEGORY: "callback"}
    )
    model_checkpoint: Union[dict, str] = field(
        default_factory=_model_checkpoint_factory, metadata={HELP: ".", CATEGORY: "callback"}
    )
    tensorboard: Union[dict, str] = field(
        default_factory=_tensorboard_factory, metadata={HELP: ".", CATEGORY: "callback"}
    )
    loss: Union[str, Dict[str, Any]] = field(
        default="CrossEntropyLoss", metadata={HELP: "pytorch loss in torch.nn package", CATEGORY: "learning"}
    )
    optimizer: Union[str, Dict[str, Any]] = field(
        default="Adam", metadata={HELP: "pytorch optimizer Class name in torch.optim package", CATEGORY: "learning"}
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            HELP: (
                "The scheduler type to use."
                "eg. `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`"
            ),
            CATEGORY: "learning"
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={HELP: "Linear warmup over warmup_ratio fraction of total steps.",
                               CATEGORY: "learning"}
    )
    warmup_steps: int = field(default=0, metadata={HELP: "Linear warmup over warmup_steps.", CATEGORY: "learning"})
    logging_dir: Optional[str] = field(default=None, metadata={HELP: "Tensorboard log dir.", CATEGORY: "logging"})
    logging_strategy: str = field(
        default="steps",
        metadata={HELP: "The logging strategy to use.", CATEGORY: "logging"},
    )
    device_str: str = field(
        default=None,
        metadata={
            HELP: (
                "None -> if there is a cuda env in the machine, it will use cuda:0, else cpu;"
                "`cpu` -> use cpu only;"
                "`cuda` -> use some gpu devices, and the using gpu count should be specified in args `n_gpu`;"
                "`cuda:2` -> use the No.2 gpu."
            ),
            CATEGORY: "device"
        }
    )
    n_gpu: int = field(default=-1, metadata={
        HELP: "should be specified when device_str is `cuda`",
        CATEGORY: "device"
    })
    sync_bn: bool = field(default=False, metadata={
        HELP: "will be work if device_str is `cuda`, the True sync_bn would make training slower but acc better.",
        CATEGORY: "device"
    })

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.device_str == "cuda" and self.n_gpu < 1:
            raise ValueError("must specify the using gpu number when device_str is `cuda`")
        self.should_save = True
        self._get_config_categories()

    @property
    def train_batch_size(self) -> int:
        assert self.batch_size > 0
        train_batch_size = self.batch_size  # * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        assert self.batch_size > 0
        if self.val_batch_size == -1:
            return self.batch_size  # * max(1, self.n_gpu)
        else:
            return self.val_batch_size

    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)

    @property
    def device(self) -> "torch.device":
        if self.device_str is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_str)
        return device

    def load_from_yaml(self, path2yaml: str = None):
        """
        Load training configuration from yaml

        :param path2yaml: the path to yaml
        :type path2yaml: str

        :example:
            >>> from towhee.trainer.training_config import TrainingConfig
            >>> conf = Path(__file__).parent / 'config.yaml'
            >>> ta = TrainingConfig()
            >>> ta.load_from_yaml(conf)
        """
        with open(path2yaml, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            for file_category in config_dict:
                if file_category not in self.config_category_set:
                    trainer_log.warning("category %s is not a attribute in TrainingConfig", file_category)
            for category in self.config_category_set:
                category_dict = config_dict.get(category, {})
                self._set_attr_from_dict(category_dict)

    def save_to_yaml(self, path2yaml: str = None):
        """
        Save training configuration to yaml

        :param path2yaml: the path to yaml
        :type path2yaml: str

        :example:
            >>> from towhee.trainer.training_config import TrainingConfig
            >>> conf = Path(__file__).parent / 'config.yaml'
            >>> ta = TrainingConfig()
            >>> ta.save_to_yaml(conf)
        """
        config_dict = {}
        for config_category in self.config_category_set:
            config_dict[config_category] = {}
        config_fields = fields(TrainingConfig)
        for config_field in config_fields:
            metadata_dict = config_field.metadata
            field_name = config_field.name
            if CATEGORY in metadata_dict:
                category = metadata_dict[CATEGORY]
                config_dict[category][field_name] = _get_attr_str(self, field_name)
            else:
                config_dict[OTHER_CATEGORY][field_name] = _get_attr_str(self, field_name)
                trainer_log.warning("metadata in self.%s has no CATEGORY", config_field.name)
        # print(config_dict)
        with open(path2yaml, "w", encoding="utf-8") as file:
            yaml.dump(config_dict, file)

    def _set_attr_from_dict(self, train_config_dict):
        for key, value in train_config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                trainer_log.warning("TrainingConfig has no attribute %s", key)

    def _get_config_categories(self):
        self.config_category_set = set()
        config_fields = fields(TrainingConfig)
        for config_field in config_fields:
            metadata_dict = config_field.metadata
            if CATEGORY in metadata_dict:
                self.config_category_set.add(metadata_dict[CATEGORY])
            else:
                self.config_category_set.add(OTHER_CATEGORY)
