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
import torchvision
from pathlib import Path
from torch import optim

from towhee.trainer.callback import TrainerControl
from towhee.trainer.training_config import TrainingConfig
from towhee.trainer.trainer import Trainer


class TestLRScheduler(unittest.TestCase):
    """
    Test lr scheduler
    """
    def test_lr_scheduler(self) -> None:
        conf = Path(__file__).parent / 'config2.yaml'
        ta = TrainingConfig()
        ta.print_steps = 2
        ta.load_from_yaml(conf)
        model = torchvision.models.resnet50(pretrained=True)
        tr = Trainer(model=model, training_config=ta)
        tr.train_dataloader = [1]
        tr.trainercontrol = TrainerControl()
        ta.tensorboard = None
        num_training_steps = 10
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        tr.optimizer = optimizer
        tr.setup_before_train(num_training_steps, 0.001)
        self.assertEqual(tr.lr_scheduler.__class__.__name__, 'StepLR')


if __name__ == '__main__':
    unittest.main(verbosity=1)
