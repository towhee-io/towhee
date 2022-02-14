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
from dataclasses import asdict, dataclass, field, fields
from enum import Enum

import torch

from typing import List, Optional
import yaml

from towhee.trainer.callback import Callback
from towhee.utils.log import trainer_log

HELP = "help"
CATEGORY = "category"
OTHER_CATEGORY = "other"


def dump_default_yaml(yaml_path):
    training_config = TrainingConfig()
    training_config.save_to_yaml(path2yaml=yaml_path)
    # print(training_config)


@dataclass
class TrainingConfig:
    """
    the training config, it can be defined in a yaml file
    """
    output_dir: str = field(
        default="./output_dir",
        metadata={HELP: "The output directory where the model predictions and checkpoints will be written.",
                  CATEGORY: "train"},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            HELP: (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            ),
            CATEGORY: "train"
        },
    )
    do_train: bool = field(default=False, metadata={HELP: "Whether to run training.", CATEGORY: "train"})
    do_eval: bool = field(default=False, metadata={HELP: "Whether to run eval on the dev set.", CATEGORY: "train"})
    do_predict: bool = field(default=False,
                             metadata={HELP: "Whether to run predictions on the test set.", CATEGORY: "train"})
    eval_strategy: str = field(
        default="no",
        metadata={HELP: "The evaluation strategy to use.", CATEGORY: "train"},
    )

    prediction_loss_only: bool = field(
        default=False,
        metadata={HELP: "When performing evaluation and predictions, only returns the loss.", CATEGORY: "train"},
    )

    batch_size: Optional[int] = field(
        default=8,
        metadata={
            HELP: "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                  "Batch size per GPU/TPU core/CPU for training.",
            CATEGORY: "train"
        }
    )
    seed: int = field(default=42,
                      metadata={HELP: "Random seed that will be set at the beginning of training.", CATEGORY: "train"})

    epoch_num: float = field(default=3.0,
                             metadata={HELP: "Total number of training epochs to perform.", CATEGORY: "train"})
    max_steps: int = field(
        default=-1,
        metadata={HELP: "If > 0: set total number of training steps to perform. Override num_train_epochs.",
                  CATEGORY: "train"},
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={HELP: "Drop the last incomplete batch if it is not divisible by the batch size.", CATEGORY: "train"}
    )

    eval_steps: int = field(default=None, metadata={HELP: "Run an evaluation every X steps.", CATEGORY: "train"})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            HELP: "Number of subprocesses to use for data loading (PyTorch only). 0 means "
                  "that the data will be loaded in the main process.",
            CATEGORY: "train"
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={HELP: "The path to a folder with a valid checkpoint for your model.", CATEGORY: "train"},
    )
    lr: float = field(default=5e-5, metadata={HELP: "The initial learning rate for AdamW.", CATEGORY: "learning"})
    weight_decay: float = field(default=0.0,
                                metadata={HELP: "Weight decay for AdamW if we apply some.", CATEGORY: "learning"})
    adam_beta1: float = field(default=0.9, metadata={HELP: "Beta1 for AdamW optimizer", CATEGORY: "learning"})
    adam_beta2: float = field(default=0.999, metadata={HELP: "Beta2 for AdamW optimizer", CATEGORY: "learning"})
    adam_epsilon: float = field(default=1e-8, metadata={HELP: "Epsilon for AdamW optimizer.", CATEGORY: "learning"})
    max_norm_grad: float = field(default=1.0, metadata={HELP: "Max gradient norm.", CATEGORY: "learning"})
    use_adafactor: bool = field(default=False, metadata={HELP: "Wif Adafactor is used.", CATEGORY: "learning"})
    metric: Optional[str] = field(
        default="Accuracy", metadata={HELP: "The metric to use to compare two different models.", CATEGORY: "metrics"}
    )

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={HELP: "Whether or not to disable the tqdm progress bars.", CATEGORY: "learning"}
    )

    label_names: Optional[List[str]] = field(
        default=None, metadata={HELP: "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    past_index: int = field(
        default=-1,
        metadata={HELP: "If >=0, uses the corresponding part of the output as the past state for next step."},
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={HELP: "Whether or not to load the best model found during training at the end of training."},
    )

    # no_cuda: bool = field(default=False, metadata={HELP: "Do not use CUDA even when it is available"})

    call_back_list: Optional[List[Callback]] = field(
        default=None, metadata={HELP: ".", CATEGORY: "callback"}
    )
    # loss: Optional[_Loss] = field(
    loss: Optional[str] = field(
        # default=nn.CrossEntropyLoss(), metadata={HELP: "pytorch loss"}
        default="CrossEntropyLoss", metadata={HELP: "pytorch loss in torch.nn package", CATEGORY: "learning"}
    )
    # optimizer: Optional[Optimizer] = field(
    optimizer: Optional[str] = field(
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
    logging_first_step: bool = field(default=False, metadata={HELP: "Log the first global_step", CATEGORY: "logging"})
    logging_steps: int = field(default=500, metadata={HELP: "Log every X updates steps.", CATEGORY: "logging"})
    logging_nan_inf_filter: str = field(default=True, metadata={HELP: "Filter nan and inf losses for logging.",
                                                                CATEGORY: "logging"})
    save_strategy: str = field(
        default="steps",
        metadata={HELP: "The checkpoint save strategy to use.", CATEGORY: "logging"},
    )
    saving_steps: int = field(default=500, metadata={HELP: "Save checkpoint every X updates steps.", CATEGORY: "logging"})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            HELP: (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            ),
            CATEGORY: "logging"
        },
    )
    saving_strategy: str = field(
        default="steps",
        metadata={HELP: "The checkpoint save strategy to use.", CATEGORY: "logging"},
    )
    saving_total_limit: Optional[int] = field(
        default=None,
        metadata={
            HELP: (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            ),
            CATEGORY: "logging"
        },
    )
    saving_on_each_node: bool = field(
        default=False,
        metadata={
            HELP: "When doing multi-node distributed training, whether to save models "
                  "and checkpoints on each node, or only on the main one",
            CATEGORY: "logging"
        },
    )
    device_str: str = field(
        default=None,
        metadata={
            HELP: (
                "None -> if there is a cuda env in the machine, it will use cuda:0, else cpu;"
                "`cpu` -> use cpu only;"
                "`cuda` -> use some gpu devices, and the using gpu count should be specified in args `n_gpu`;"
                "`cuda:2` -> use the No.2 gpu."
            )
        }
    )
    n_gpu: int = field(default=-1, metadata={
        HELP: "should be specified when device_str is `cuda`"
    })
    sync_bn: bool = field(default=True, metadata={
        HELP: "will be work if device_str is `cuda`, the True sync_bn would make training a little slower"
    })

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        # if self.disable_tqdm is None:
        #     self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
        if self.device_str == "cuda" and self.n_gpu < 1:
            raise ValueError("must specify the using gpu number when device_str is `cuda`")
        self.should_save = True

        self._get_config_categories()

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training.
        """
        train_batch_size = self.batch_size * max(1, self.n_gpu)
        return train_batch_size

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
    def device(self) -> "torch.device":
        if self.device_str is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_str)
        return device

    def load_from_yaml(self, path2yaml: str = None):
        with open(path2yaml, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            train_config_dict = config_dict["train"]["args"]
            learning_config_dict = config_dict["learning"]["args"]
            callback_config_dict = config_dict["callback"]["args"]
            metrics_config_dict = config_dict["metrics"]["args"]
            logging_config_dict = config_dict["logging"]["args"]
            self._set_attr_from_dict(train_config_dict)
            self._set_attr_from_dict(learning_config_dict)
            self._set_attr_from_dict(callback_config_dict)
            self._set_attr_from_dict(metrics_config_dict)
            self._set_attr_from_dict(logging_config_dict)

    def save_to_yaml(self, path2yaml: str = None):
        config_dict = {}
        for config_category in self.config_category_set:
            config_dict[config_category] = {"args": {}}
        config_fields = fields(TrainingConfig)
        for config_field in config_fields:
            metadata_dict = config_field.metadata
            field_name = config_field.name
            if CATEGORY in metadata_dict:
                category = metadata_dict[CATEGORY]
                config_dict[category]["args"][field_name] = getattr(self, field_name)
            else:
                config_dict[OTHER_CATEGORY]["args"][field_name] = getattr(self, field_name)
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
