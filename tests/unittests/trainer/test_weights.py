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

import unittest
import torch
import torchvision
from pathlib import Path

from towhee.trainer.utils.trainer_utils import STATE_CHECKPOINT_NAME

from towhee.trainer.training_config import TrainingConfig

from towhee.operator import NNOperator
from towhee.trainer.modelcard import ModelCard


class MockOperator(NNOperator):
    def __init__(self, model_name='resnet50', framework='pytorch'):
        super().__init__(framework=framework)
        self.model_name = model_name
        self.model = torchvision.models.resnet50(pretrained=True)
    def __call__(self, x):
        return self.model(x)
    def get_model(self):
        return self.model

class TestWeights(unittest.TestCase):
    """
    Test save_weights
    """
    op = MockOperator()
    x = torch.rand([1, 3, 224, 224])
    training_config = TrainingConfig(
        output_dir='./temp_output',
        overwrite_output_dir=True,
        epoch_num=2,
        batch_size=8,
    )
    model_card = ModelCard(model_overview='efficientnet test modelcard',
                           datasets='use efficientnet test data')

    def test_with_trainer(self):
        self.op.save('./test_save')
        filepath = Path('./test_save').joinpath(STATE_CHECKPOINT_NAME)
        self.assertTrue(Path(filepath).is_file())

        with self.assertRaises(FileExistsError):
            self.op.save('./test_save', overwrite=False)

        out1 = self.op(self.x)
        self.op.load('./test_save')
        out2 = self.op(self.x)
        self.assertTrue((out1 == out2).all())


if __name__ == '__main__':
    unittest.main()
