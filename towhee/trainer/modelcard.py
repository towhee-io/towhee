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


import os
from dataclasses import dataclass
from typing import Union, Dict, Any, Optional, List

# from towhee.utils.log import trainer_log
from towhee.trainer.training_config import TrainingConfig

MODEL_CARD_NAME = "README.md"


@dataclass
class ModelCard:
    """
    Utilities to generate and save model card. Recommended attributes from https://arxiv.org/abs/1810.03993 (see papers)
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet

    Args:
        model_name (`Optional[str]`):
            model name
        model_architecture (`Optional[str]`):
            model structure
        model_overview (`Optional[str] = None`):
        language (`Optional[Union[str, List[str]]]`):
            language
        tags (`Optional[Union[str, List[str]]]`):
            tags
        tasks (`Optional[Union[str, List[str]]]`):
            model tasks (eg. classification, prediction, etc.)
        datasets (`Optional[Union[str, List[str]]]`):
            datasets used to train/test the model
        datasets_tags (`Optional[Union[str, List[str]]]`):
            tags of datasets
        dataset_args (`Optional[Union[str, List[str]]]`):
            arguments of dataset
        eval_results (`Optional[Dict[str, float]]`):
            evaluation results recorded
        eval_lines (`Optional[List[str]]`):
            evaluation baselines
        training_summary (`Optional[Dict[str, Any]]`):
            training summary include training information
        training_config (`Optional[TrainingConfig]`):
            training configurations
        source (`Optional[str]`):
            source of model card (default = "trainer")

    Example:
        >>> from towhee.trainer.modelcard import ModelCard
        >>> model_card = ModelCard(model_name='test')
        >>> # Print out model name stored in model card
        >>> model_card.model_name
        'test'
        >>> # Save model card to "path/to/my_dir" as README.md
        >>> model_card.save_model_card('/path/to/my_dir')
        >>> # Save model card as "/path/to/my_dir/model_card.md"
        >>> model_card.save_model_card('/path/to/my_dir/model_card.md')
    """
    model_name: Optional[str] = None
    model_architecture: Optional[str] = None
    model_overview: Optional[str] = None
    language: Optional[Union[str, List[str]]] = None
    # license: Optional[str] = None
    tags: Optional[Union[str, List[str]]] = None
    tasks: Optional[Union[str, List[str]]] = None
    datasets: Optional[Union[str, List[str]]] = None
    datasets_tags: Optional[Union[str, List[str]]] = None
    dataset_args: Optional[Union[str, List[str]]] = None
    eval_results: Optional[Dict[str, float]] = None
    eval_lines: Optional[List[str]] = None
    training_summary: Optional[Dict[str, Any]] = None
    training_config: Optional[TrainingConfig] = None
    source: Optional[str] = "trainer"

    def __post_init__(self):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self._to_json_string())

    def save_model_card(self, save_directory_or_file):
        """
        Write model card to the given filepath or directory

        Args:
            save_directory_or_file (`str`):
                file path or directory to write and save model card.
        """
        model_card = f"# {self.model_name}"
        model_card += f"\n\n## Model overview\n{self.model_overview}\n"
        model_card += f"\n\n### Model architecture\n{self.model_architecture}\n"

        model_card += "\n\n## Dataset"
        if self.datasets is None:
            model_card += "\nDataset unknown.\n"
        else:
            if isinstance(self.datasets, str):
                model_card += f"\nUsing the {self.datasets} as dataset.\n"
            elif isinstance(self.dataset, (tuple, list)) and len(self.datasets) == 1:
                model_card += f"\nUsing the {self.datasets} as dataset.\n"
            else:
                model_card += (
                        "\n, ".join(
                            [f"Using the {dataset}" for dataset in self.datasets[:-1]])
                        + f" and the {self.datasets[-1]} dataset.\n"
                )

        model_card += "\n\n## Training configurations\n"
        if self.training_config is not None:
            if self.training_config.optimizer is not None:
                if isinstance(self.training_config.optimizer, str):
                    model_card += f"\nOptimizer is  {self.training_config.optimizer}.\n"
                else:
                    model_card += "\n".join([f"- {name}: "
                                             f"{value}" for name, value in self.training_config.optimizer.items()])
                    model_card += "\n"
            if self.training_config.lr_scheduler_type is not None:
                model_card += f"\nThe scheduler type is {self.training_config.lr_scheduler_type}.\n"
            if self.training_config.warmup_ratio is not None:
                model_card += f"\nThe warmup_ratio is {self.training_config.warmup_ratio}.\n"
            if self.training_config.warmup_steps is not None:
                model_card += f"\nThe warmup_steps is {self.training_config.warmup_steps}.\n"
            if self.training_config.lr is not None:
                model_card += f"\nLearning rate is {self.training_config.lr}.\n"
            else:
                model_card += "\nLearning rate is needed.\n"
            # if self.training_config.weight_decay is not None:
            #     model_card += f"\nWeight decay is {self.training_config.weight_decay}."
            # if self.training_config.adam_beta1 is not None:
            #     model_card += f"\nBeta1 for AdamW optimizer is {self.training_config.adam_beta1}."
            # if self.training_config.adam_beta2 is not None:
            #     model_card += f"\nBeta2 for AdamW optimizer is {self.training_config.adam_beta2}."
            # if self.training_config.adam_epsilon is not None:
            #     model_card += f"\nEpsilon for AdamW optimizer is {self.training_config.adam_epsilon}."
            if self.training_config.metric is not None:
                model_card += f"\nMetric is {self.training_config.metric}.\n"
            #
            # if self.training_config.max_norm_grad is not None:
            #     model_card += f"\nMax gradient norm is {self.training_config.max_norm_grad}."
            if self.training_config.batch_size is not None:
                model_card += f"\nBatch size is {self.training_config.batch_size}.\n"
            if self.training_config.seed is not None:
                model_card += f"\nRandom seed is {self.training_config.seed}.\n"
            if self.training_config.epoch_num is not None:
                model_card += f"\nTraining epochs is {self.training_config.epoch_num}.\n"
            # if self.training_config.n_gpu is not None:
            #     model_card += f"\nNumber of GPUs is {self.training_config.n_gpu}.\n"
        else:
            model_card += "\nTraining configurations is needed.\n"

        model_card += "\n\n## Training summary\n"
        if self.training_summary is not None:
            model_card += "\n"
            model_card += "\n".join([f"- {name}: {value}" for name, value in self.training_summary.items()])
            model_card += "\n"
        else:
            model_card += "\nTraining summary is needed.\n"

        if os.path.isdir(save_directory_or_file):
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_card_file = os.path.join(save_directory_or_file, MODEL_CARD_NAME)
        else:
            output_model_card_file = save_directory_or_file

        with open(output_model_card_file, "w", encoding="utf-8") as f:
            f.write(model_card)

    def _to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        pass

    def _to_json_string(self):
        """Serializes this instance to a JSON string."""
        pass

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        pass

    @staticmethod
    def load_from_file(file_path):
        pass

