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
import json
import copy

from towhee.utils.log import trainer_log

MODEL_CARD_NAME = "model_card.json"

class ModelCard():
    """
    Recommended attributes from https://arxiv.org/abs/1810.03993 (see papers)
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet
    """
    def __init__(self, **kwargs):
        self.model_details = kwargs.pop("model_details", {})
        self.intended_use = kwargs.pop("intended_use", {})
        self.factors = kwargs.pop("factors", {})
        self.metrics = kwargs.pop("metrics", {})
        self.evaluation_data = kwargs.pop("evaluation_data", {})
        self.training_data = kwargs.pop("training_data", {})
        self.quantitative_analyses = kwargs.pop("quantitative_analyses", {})
        self.ethical_considerations = kwargs.pop("ethical_considerations", {})
        self.caveats_and_recommendations = kwargs.pop("caveats_and_recommendations", {})

        # Open additional attributes
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                trainer_log.error("Can't set %s with value %s for %s", key, value, self)
                # todo:test this line
                raise err

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
        if os.path.isdir(save_directory_or_file):
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_card_file = os.path.join(save_directory_or_file, MODEL_CARD_NAME)
        else:
            output_model_card_file = save_directory_or_file

        self._to_json_file(output_model_card_file)
        trainer_log.info("Model card saved in %s", output_model_card_file) #todo:logger.info can't show

    def _to_json_file(self, json_file_path):
        """Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self._to_json_string())

    def _to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    @staticmethod
    def load_from_file(file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                model_card_dict = json.loads(f.read())
                return ModelCard(**model_card_dict)
        raise FileNotFoundError

# if __name__ == '__main__':  # todo: delete
#     # mc = ModelCard(model_details="abc", factors=1234, type='aaabbb')
#     mc = ModelCard.load_from_file('./aaa.txt')
#     print(mc)
#     print(mc.to_dict())
#     # print(mc.to_json_string())
#     # mc.save_model_card('./aaa.txt')
#     print(1)
