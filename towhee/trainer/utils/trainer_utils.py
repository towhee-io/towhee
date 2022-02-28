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
"""
Utilities for the Trainer.
"""

import random
from typing import NamedTuple, Any, Generator, Callable
from collections.abc import Mapping
from enum import Enum
import numpy as np
import torch
import os
from pathlib import Path
import torch.distributed as dist


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_last_checkpoint(out_dir: str):
    checkpoints = [path for path in Path(out_dir).iterdir() if path.is_dir()]
    if len(checkpoints) == 0:
        raise ValueError(f"No checkpoints found at {out_dir}.")
    return max(checkpoints, key=os.path.getmtime).resolve()


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


STATE_CHECKPOINT_NAME = "state.pt"
MODEL_NAME = "model.pth"


# _re_checkpoint = re.compile(r"^" + CHECKPOINT_NAME + r"\-(\d+)$")

class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # one gpu
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def honor_type(obj, generator: Generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    # There is no direct check whether an object if of type namedtuple sadly, this is a workaround.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # Can instantiate a namedtuple from a generator directly, contrary to a tuple/list.
        return type(obj)(*list(generator))
    return type(obj)(generator)

def is_torch_tensor(tensor: Any):
    return isinstance(tensor, torch.Tensor)

def recursively_apply(func: Callable, data: Any, *args, test_type: Callable=is_torch_tensor, error_on_other_type: bool=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (:obj:`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of :obj:`main_type`):
            The data on which to apply :obj:`func`
        *args:
            Positional arguments that will be passed to :obj:`func` when applied on the unpacked data.
        main_type (:obj:`type`, `optional`, defaults to :obj:`torch.Tensor`):
            The base type of the objects to which apply :obj:`func`.
        error_on_other_type (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return an error or not if after unpacking :obj:`data`, we get on an object that is not of type
            :obj:`main_type`. If :obj:`False`, the function will leave objects of types different than :obj:`main_type`
            unchanged.
        **kwargs:
            Keyword arguments that will be passed to :obj:`func` when applied on the unpacked data.

    Returns:
        The same data structure as :obj:`data` with :obj:`func` applied to every object of type :obj:`main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def send_to_device(tensor: Any, device: torch.device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    Borrowed from huggingface/accelerate.
    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """

    def _send_to_device(t, device):
        return t.to(device)

    def _has_to_method(t):
        return hasattr(t, "to")

    return recursively_apply(_send_to_device, tensor, device, test_type=_has_to_method)
