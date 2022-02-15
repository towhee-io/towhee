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

import unittest
from pathlib import Path

from towhee.trainer.training_config import TrainingConfig


class TestTrainConfig(unittest.TestCase):
    """
    Test TrainingConfig init
    """
    def test_trainconfig_init(self) -> None:
        conf = Path(__file__).parent / 'config.yaml'
        ta = TrainingConfig()
        ta.load_from_yaml(conf)
        self.assertEqual(ta.epoch_num, 2)


if __name__ == '__main__':
    unittest.main(verbosity=1)
