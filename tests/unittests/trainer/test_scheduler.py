# Copyright 2020-present the HuggingFace Inc. team and 2021 Zilliz.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import tempfile
import unittest
import torch
from torch import nn

from towhee.trainer.optimization.adafactor import Adafactor
from towhee.trainer.optimization.adamw import AdamW
from towhee.trainer.scheduler import (
    configure_constant_scheduler,
    configure_constant_scheduler_with_warmup,
    configure_linear_scheduler_with_warmup,
    configure_cosine_scheduler_with_warmup,
    configure_cosine_with_hard_restarts_scheduler_with_warmup,
    configure_polynomial_decay_scheduler_with_warmup
)


def unwrap_scheduler(scheduler, num_steps=10):
    lr_sch = []
    for _ in range(num_steps):
        lr_sch.append(scheduler.get_lr()[0])
        scheduler.step()
    return lr_sch


def unwrap_and_save_reload_scheduler(scheduler, num_steps=10):
    lr_sch = []
    for step in range(num_steps):
        lr_sch.append(scheduler.get_lr()[0])
        scheduler.step()
        if step == num_steps // 2:
            with tempfile.TemporaryDirectory() as tmpdirname:
                file_name = os.path.join(tmpdirname, 'schedule.bin')
                torch.save(scheduler.state_dict(), file_name)

                state_dict = torch.load(file_name)
                scheduler.load_state_dict(state_dict)
    return lr_sch


class ScheduleInitTest(unittest.TestCase):
    mdl = nn.Linear(50, 50)
    optimizer = AdamW(mdl.parameters(), lr=10.0)
    num_steps = 10

    def assertListAlmostEqual(self, list1, list2, tol, msg=None):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol, msg=msg)

    def test_schedulers(self):
        common_kwargs = {'num_warmup_steps': 2, 'num_training_steps': 10}
        # schedulers doct format
        # function: (sched_args_dict, expected_learning_rates)
        scheds = {
            configure_constant_scheduler: ({}, [10.0] * self.num_steps),
            configure_constant_scheduler_with_warmup: (
                {'num_warmup_steps': 4},
                [0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ),
            configure_linear_scheduler_with_warmup: (
                {**common_kwargs},
                [0.0, 5.0, 10.0, 8.75, 7.5, 6.25, 5.0, 3.75, 2.5, 1.25],
            ),
            configure_cosine_scheduler_with_warmup: (
                {**common_kwargs},
                [0.0, 5.0, 10.0, 9.61, 8.53, 6.91, 5.0, 3.08, 1.46, 0.38],
            ),
            configure_cosine_with_hard_restarts_scheduler_with_warmup: (
                {**common_kwargs, 'num_cycles': 2},
                [0.0, 5.0, 10.0, 8.53, 5.0, 1.46, 10.0, 8.53, 5.0, 1.46],
            ),
            configure_polynomial_decay_scheduler_with_warmup: (
                {**common_kwargs, 'power': 2.0, 'lr_end': 1e-7},
                [0.0, 5.0, 10.0, 7.656, 5.625, 3.906, 2.5, 1.406, 0.625, 0.156],
            ),
        }

        for scheduler_func, data in scheds.items():
            kwargs, expected_learning_rates = data

            scheduler = scheduler_func(self.optimizer, **kwargs)
            self.assertEqual(len([scheduler.get_lr()[0]]), 1)
            lr_sch_1 = unwrap_scheduler(scheduler, self.num_steps)
            self.assertListAlmostEqual(
                lr_sch_1,
                expected_learning_rates,
                tol=1e-2,
                msg=f'failed for {scheduler_func} in normal scheduler',
            )

            scheduler = scheduler_func(self.optimizer, **kwargs)
            lr_sch_2 = unwrap_and_save_reload_scheduler(scheduler, self.num_steps)
            self.assertListEqual(lr_sch_1, lr_sch_2, msg=f'failed for {scheduler_func} in save and reload')


class OptimizationTest(unittest.TestCase):
    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def test_adam_w(self):
        w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
        target = torch.tensor([0.4, 0.2, -0.5])
        criterion = nn.MSELoss()
        # No warmup, constant schedule, no gradient clipping
        optimizer = AdamW(params=[w], lr=2e-1, weight_decay=0.0)
        for _ in range(100):
            loss = criterion(w, target)
            loss.backward()
            optimizer.step()
            w.grad.detach_()  # No zero_grad() function on simple tensors. we do it ourselves.
            w.grad.zero_()
        self.assertListAlmostEqual(w.tolist(), [0.4, 0.2, -0.5], tol=1e-2)

    def test_adafactor(self):
        w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
        target = torch.tensor([0.4, 0.2, -0.5])
        criterion = nn.MSELoss()
        # No warmup, constant schedule, no gradient clipping
        optimizer = Adafactor(
            params=[w],
            lr=1e-2,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        for _ in range(1000):
            loss = criterion(w, target)
            loss.backward()
            optimizer.step()
            w.grad.detach_()  # No zero_grad() function on simple tensors. we do it ourselves.
            w.grad.zero_()
        self.assertListAlmostEqual(w.tolist(), [0.4, 0.2, -0.5], tol=1e-2)


if __name__ == '__main__':
    unittest.main()
