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
from typing import NamedTuple
from enum import Enum
import numpy as np
import torch
import os
from pathlib import Path


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


CHECKPOINT_NAME = "checkpoint.pt"


# _re_checkpoint = re.compile(r"^" + CHECKPOINT_NAME + r"\-(\d+)$")
class EvalStrategyType(Enum):
    EPOCH = "epoch"
    STEP = "step"
    NO = "no"

class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
