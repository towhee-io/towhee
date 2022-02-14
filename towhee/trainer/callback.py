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

from typing import Dict, Tuple, List

from torch import nn
from torch.optim import Optimizer

__all__ = [
    'Callback',
    'CallbackList',
    'TrainerControl'
]

def configure_callbacklist(configs: Dict) -> None:
    # TODO: construct the callbacks from configure file.
    raise NotImplementedError

class TrainerControl:
    """
    `TrainerControl` defines a set of current control status which trainer
    can get and take the corresponding action. It can be used by customized
    `Callback` to interfere the trainer.

    Args:
        should_training_stop: (`bool`)
            whether or not training should be interrupted.
        should_epoch_stop: (`bool`)
            whether or not current training epoch should be interrupted.
        should_save: (`bool`)
            whether or not trainer should save current model.
        should_evaluate: (`bool`)
            whether or not trainer should evaluate current model.
        should_log: (`bool`)
            whether or not trainer should report the log.
    """

    def __init__(self,
                should_training_stop = False,
                should_epoch_stop = False,
                should_save = False,
                should_evaluate = False,
                should_log = False):
        self.should_training_stop = should_training_stop
        self.should_epoch_stop = should_epoch_stop
        self.should_save = should_save
        self.should_evaluate = should_evaluate
        self.should_log = should_log

class Callback:
    """
    `Callback` defines a set of functions which will be called in the training
    process. Customized `Callback` could inherent the base `Callback` and
    overwrite its methods to control the training process or handle the training
    information.
    """

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.trainercontrol = None

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def set_trainercontrol(self, trainercontrol: TrainerControl) -> None:
        self.trainercontrol = trainercontrol

    def on_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        pass

    def on_batch_end(self, batch: Tuple, logs: Dict) -> None:
        pass

    def on_epoch_begin(self, epochs: int, logs: Dict) -> None:
        pass

    def on_epoch_end(self, epochs: int, logs: Dict) -> None:
        pass

    def on_train_begin(self, logs: Dict) -> None:
        pass

    def on_train_end(self, logs: Dict) -> None:
        pass

    def on_train_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        self.on_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        self.on_batch_end(batch, logs)

    def on_eval_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        self.on_batch_begin(batch, logs)

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        self.on_batch_end(batch, logs)

    def on_eval_begin(self, logs: Dict) -> Dict:
        pass

    def on_eval_end(self, logs: Dict) -> Dict:
        pass

class CallbackList:
    """
    `CallbackList` aggregate multiple `Callback` in the same object. Invoke the
    callbacks of `CallbackList` will invoke corresponding callback in each
    'Callback' in the FIFO sequential order.
    """
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = []
        if callbacks is not None:
            self.callbacks.extend(callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def __getitem__(self, idx: int) -> Callback:
        return self.callbacks[idx]

    def __repr__(self):
        callback_desc = ''
        for cb in self.callbacks:
            callback_desc += cb.__repr__() + ','
        return 'towhee.trainer.CallbackList([{}])'.format(callback_desc)

    def set_model(self, model: nn.Module):
        self.model = model
        for cb in self.callbacks:
            cb.set_model(model)

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        for cb in self.callbacks:
            cb.set_optimizer(optimizer)

    def set_trainercontrol(self, trainercontrol: TrainerControl):
        self.trainercontrol = TrainerControl
        for cb in self.callbacks:
            cb.set_trainercontrol(trainercontrol)

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def pop_callback(self, callback: Callback):
        self.callbacks.remove(callback)

    def on_batch_begin(self, epochs: int, batch: Tuple, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(epochs, batch, logs)

    def on_batch_end(self, epochs: int, batch: Tuple, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(epochs, batch, logs)

    def on_epoch_begin(self, epochs: int, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epochs, logs)

    def on_epoch_end(self, epochs: int, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epochs, logs)

    def on_train_begin(self, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_train_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs)

    def on_eval_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_eval_batch_begin(batch, logs)

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_eval_batch_end(batch, logs)

    def on_eval_begin(self, logs: Dict) -> Dict:
        for cb in self.callbacks:
            cb.on_eval_begin(logs)

    def on_eval_end(self, logs: Dict) -> Dict:
        for cb in self.callbacks:
            cb.on_eval_end(logs)
