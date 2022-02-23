# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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


"""Scheduler utilities for pytorch optimization."""


import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


scheduler_list = [
    'constant',
    'constant_with_warmup',
    'linear',
    'cosine',
    'cosine_with_restarts',
    'polynomial'
]


def check_scheduler(scheduler_type: str) -> bool:
    """
    Check if the scheduler type is supported.

    :param scheduler_type: the type of the scheduler
    :type scheduler_type: str

    :return bool:
        if the scheduler type is supported

    :example:
        >>> from towhee.trainer.scheduler import check_scheduler
        >>> check_scheduler('cosine')
    """
    if scheduler_list.count(scheduler_type) == 0:
        return False
    else:
        return True


def configure_constant_scheduler(optimizer: Optimizer, last_epoch: int = -1):
    """
    Return a scheduler with a constant learning rate, using the learning rate set in optimizer.

    :param optimizer: the optimizer for which to schedule the learning rate
    :type optimizer: Optimizer

    :param last_epoch: the last epoch when resuming training
    :type last_epoch: int

    :return LambdaLR:
        a constant scheduler

    :example:
        >>> from towhee.trainer.scheduler import configure_constant_scheduler
        >>> from torch import nn
        >>> from towhee.trainer.optimization.adamw import AdamW
        >>> mdl = nn.Linear(50, 50)
        >>> optimizer = AdamW(mdl.parameters(), lr=10.0)
        >>> configure_constant_scheduler(optimizer)
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def configure_constant_scheduler_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Return a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    :param optimizer: the optimizer for which to schedule the learning rate
    :type optimizer: Optimizer

    :param num_warmup_steps: warmup steps
    :type num_warmup_steps: int

    :param last_epoch: the last epoch when training is resumed
    :type last_epoch: int

    :return LambdaLR:
        a constant scheduler with warmup

    :example:
        >>> from towhee.trainer.scheduler import configure_constant_scheduler_with_warmup
        >>> from torch import nn
        >>> from towhee.trainer.optimization.adamw import AdamW
        >>> mdl = nn.Linear(50, 50)
        >>> num_warmup_steps = 4
        >>> num_training_steps = 10
        >>> optimizer = AdamW(mdl.parameters(), lr=10.0)
        >>> configure_constant_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def configure_linear_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Return a scheduler with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    :param optimizer: the optimizer for which to schedule the learning rate
    :type optimizer: Optimizer

    :param num_warmup_steps: warmup steps
    :type num_warmup_steps: int

    :param num_training_steps: training steps
    :type num_training_steps: int

    :param last_epoch: the last epoch when training is resumed
    :type last_epoch: int

    :return LambdaLR:
        a linear scheduler with warmup

    :example:
        >>> from towhee.trainer.scheduler import configure_constant_scheduler_with_warmup
        >>> from torch import nn
        >>> from towhee.trainer.optimization.adamw import AdamW
        >>> num_warmup_steps = 4
        >>> num_training_steps = 10
        >>> mdl = nn.Linear(50, 50)
        >>> optimizer = AdamW(mdl.parameters(), lr=10.0)
        >>> configure_linear_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def configure_cosine_scheduler_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Return a scheduler with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    :param optimizer: the optimizer for which to schedule the learning rate
    :type optimizer: Optimizer

    :param num_warmup_steps: warmup steps
    :type num_warmup_steps: int

    :param num_training_steps: training steps
    :type num_training_steps: int

    :param num_cycles: the number of periods in te cosine scheduler
    :type num_cycles: int

    :param last_epoch: the last epoch when training is resumed
    :type last_epoch: int

    :return LambdaLR:
        a cosine scheduler with warmup

    :example:
        >>> from towhee.trainer.scheduler import configure_constant_scheduler_with_warmup
        >>> from torch import nn
        >>> from towhee.trainer.optimization.adamw import AdamW
        >>> num_warmup_steps = 4
        >>> num_training_steps = 10
        >>> num_cycles = 2
        >>> mdl = nn.Linear(50, 50)
        >>> optimizer = AdamW(mdl.parameters(), lr=10.0)
        >>> configure_cosine_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def configure_cosine_with_hard_restarts_scheduler_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Return a scheduler with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    :param optimizer: the optimizer for which to schedule the learning rate
    :type optimizer: Optimizer

    :param num_warmup_steps: warmup steps
    :type num_warmup_steps: int

    :param num_training_steps: training steps
    :type num_training_steps: int

    :param num_cycles: the number of periods in te cosine scheduler
    :type num_cycles: int

    :param last_epoch: the last epoch when training is resumed
    :type last_epoch: int

    :return LambdaLR:
        a cosine with hard restarts scheduler with warmup

    :example:
        >>> from towhee.trainer.scheduler import configure_cosine_with_hard_restarts_scheduler_with_warmup
        >>> from torch import nn
        >>> from towhee.trainer.optimization.adamw import AdamW
        >>> num_warmup_steps = 4
        >>> num_training_steps = 10
        >>> num_cycles = 2
        >>> mdl = nn.Linear(50, 50)
        >>> optimizer = AdamW(mdl.parameters(), lr=10.0)
        >>> configure_cosine_with_hard_restarts_scheduler_with_warmup(optimizer, num_warmup_steps,
        num_training_steps, num_cycles)
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def configure_polynomial_decay_scheduler_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Return a scheduler with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    :param optimizer: the optimizer for which to schedule the learning rate
    :type optimizer: Optimizer

    :param num_warmup_steps: warmup steps
    :type num_warmup_steps: int

    :param num_training_steps: training steps
    :type num_training_steps: int

    :param lr_end: the end LR
    :type lr_end: float

    :param power: power factor
    :type power: float

    :param last_epoch: the last epoch when training is resumed
    :type last_epoch: int

    :return LambdaLR:
        a polynomial decay scheduler with warmup

    :example:
        >>> from towhee.trainer.scheduler import configure_cosine_with_hard_restarts_scheduler_with_warmup
        >>> from torch import nn
        >>> from towhee.trainer.optimization.adamw import AdamW
        >>> num_warmup_steps = 4
        >>> num_training_steps = 10
        >>> power = 2.0
        >>> lr_end = 1e-7
        >>> mdl = nn.Linear(50, 50)
        >>> optimizer = AdamW(mdl.parameters(), lr=10.0)
        >>> configure_cosine_with_hard_restarts_scheduler_with_warmup(optimizer, num_warmup_steps,
        num_training_steps, lr_end, power)
    """
    lr_init = optimizer.defaults['lr']
    assert lr_init > lr_end, f'lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})'

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# class AdafactorScheduler(LambdaLR):
#     """
#     Adafactor scheduler.
#
#     It returns ``initial_lr`` during startup and the actual ``lr`` during stepping.
#     """
#
#     def __init__(self, optimizer: Optimizer, initial_lr=0.0):
#         def lr_lambda(_):
#             return initial_lr
#
#         for group in optimizer.param_groups:
#             group['initial_lr'] = initial_lr
#         super().__init__(optimizer, lr_lambda)
#         for group in optimizer.param_groups:
#             del group['initial_lr']
#
#     def get_lr(self):
#         opt = self.optimizer
#         lrs = [
#             opt._get_lr(group, opt.state[group['params'][0]])
#             for group in opt.param_groups
#             if group['params'][0].grad is not None
#         ]
#         if len(lrs) == 0:
#             lrs = self.base_lrs  # if called before stepping
#         return lrs
#
#
# def congigure_adafactor_scheduler(optimizer: Optimizer, initial_lr=0.0):
#     """
#     Return an Adafactor scheduler.
#
#     Args:
#         optimizer:
#             The optimizer to be scheduled.
#         initial_lr:
#             Initial lr.
#
#     Return:
#         An Adafactor scheduler.
#     """
#     return AdafactorScheduler(optimizer, initial_lr)
