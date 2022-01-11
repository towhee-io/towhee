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

from torch import optim
from trainer.callback import DefaultFlowCallback, CallbackHandler, ProgressCallback

DEFAULT_CALLBACKS = [DefaultFlowCallback]


class TrainerCallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torchvision.models.resnet50(pretrained=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def check_callbacks_equality(self, cbs1, cbs2):
        self.assertEqual(len(cbs1), len(cbs2))

        # Order doesn't matter
        cbs1 = list(sorted(cbs1, key=lambda cb: cb.__name__ if isinstance(cb, type) else cb.__class__.__name__))
        cbs2 = list(sorted(cbs2, key=lambda cb: cb.__name__ if isinstance(cb, type) else cb.__class__.__name__))
        for cb1, cb2 in zip(cbs1, cbs2):
            if isinstance(cb1, type) and isinstance(cb2, type):
                self.assertEqual(cb1, cb2)
            elif isinstance(cb1, type) and not isinstance(cb2, type):
                self.assertEqual(cb1, cb2.__class__)
            elif not isinstance(cb1, type) and isinstance(cb2, type):
                self.assertEqual(cb1.__class__, cb2)
            else:
                self.assertEqual(cb1, cb2)

    def test_add_callback(self) -> None:
        callbacks = DEFAULT_CALLBACKS.copy() + [ProgressCallback]
        callback_handler = CallbackHandler(
            callbacks, self.model, self.optimizer, self.lr_scheduler
        )
        expected_callbacks = DEFAULT_CALLBACKS.copy() + [ProgressCallback]
        self.check_callbacks_equality(callback_handler.callbacks, expected_callbacks)


if __name__ == '__main__':
    unittest.main(verbosity=1)
