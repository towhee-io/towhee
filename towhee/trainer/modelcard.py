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
    Recommended attributes from https://arxiv.org/abs/1810.03993 (see papers)
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet
    """
    model_name: Optional[str] = None
    model_architecture: Optional[str] = None
    model_overview: Optional[str] = None
    language: Optional[Union[str, List[str]]] = None
    license: Optional[str] = None
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
        Save a model card as a json file,
        save_directory_or_file can be directory or file path,
        if it's a directory, default name is 'modelcard.json,
        if there exist a same named file, it will be overwritten
        """
        model_card = f"# {self.model_name}\n\n"
        model_card += f"## model overview\n{self.model_overview}\n\n"
        model_card += f"### model architecture\n{self.model_architecture}\n\n"
        model_card += "## dataset\n"
        if self.datasets is None:
            model_card += "Dataset unknown.\n"
        else:
            if isinstance(self.datasets, str):
                model_card += f"Using the {self.datasets} as dataset."
            elif isinstance(self.dataset, (tuple, list)) and len(self.datasets) == 1:
                model_card += f"Using the {self.datasets} as dataset."
            else:
                model_card += (
                        ", ".join(
                            [f"Using the {dataset}" for dataset in self.datasets[:-1]])
                        + f" and the {self.datasets[-1]} dataset."
                )
        model_card += "\n## Training configurations\n"
        if self.training_config is not None:
            if self.training_config.optimizer is not None:
                model_card += f"\nOptimizer is  {self.training_config.optimizer}."
                if isinstance(self.training_config.optimizer, str):
                    model_card += f"\nOptimizer is  {self.training_config.optimizer}."
                else:
                    model_card += "\n".join([f"- {name}: "
                                             f"{value}" for name, value in self.training_config.optimizer.items()])
                model_card += "\n"
            if self.training_config.lr_scheduler_type is not None:
                model_card += f"\nThe scheduler type is {self.training_config.lr_scheduler_type}."
            if self.training_config.warmup_ratio is not None:
                model_card += f"\nThe warmup_ratio is {self.training_config.warmup_ratio}."
            if self.training_config.warmup_steps is not None:
                model_card += f"\nThe warmup_steps is {self.training_config.warmup_steps}."
            if self.training_config.lr is not None:
                model_card += f"\nLearining reate is {self.training_config.lr}."
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
                model_card += f"\nMetric is {self.training_config.metric}."
            #
            # if self.training_config.max_norm_grad is not None:
            #     model_card += f"\nMax gradient norm is {self.training_config.max_norm_grad}."
            if self.training_config.batch_size is not None:
                model_card += f"\nBatch size is {self.training_config.batch_size}."
            if self.training_config.seed is not None:
                model_card += f"\nRandom seed is {self.training_config.seed}."
            if self.training_config.epoch_num is not None:
                model_card += f"\nTraining epochs is {self.training_config.epoch_num}."
            if self.training_config.n_gpu is not None:
                model_card += f"\nNumber of GPUs is {self.training_config.n_gpu}."
        else:
            model_card += "\nTraining configurations is needed.\n"


        model_card += "\n## Training summary\n"
        if self.training_summary is not None:
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

# if __name__ == '__main__':  # todo: delete
#     # mc = ModelCard(model_details="abc", factors=1234, type='aaabbb')
#     mc = ModelCard.load_from_file('./aaa.txt')
#     print(mc)
#     print(mc.to_dict())
#     # print(mc.to_json_string())
#     # mc.save_model_card('./aaa.txt')
#     print(1)
