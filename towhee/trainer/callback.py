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

from typing import Dict, Tuple, List

import numpy as np
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

class EarlyStopping(Callback):
    """
    EarlyStopping
    """
    def __init__(self,
                 monitor,
                 min_delta = 0,
                 patience = 0,
                 mode = 'max',
                 baseline = None
                 ):
        #super(EarlyStopping, self).__init__()
        super(EarlyStopping).__init__()
        self.monitor = monitor
        self.patience = patience
        #self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        assert mode in ['max', 'min']

        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epochs, logs = None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        #if self.restore_best_weights and self.best_weights is None:
        #    self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epochs
           # if self.restore_best_weights:
           #     self.best_weights = self.model.get_weights()
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience and epochs > 0:
            self.stopped_epoch = epochs

    def on_train_end(self, logs = None):
        #TODO
        pass

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)

        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

class ModelCheckpoint(Callback):
    """
    ModelCheckpoint
    """
    def __init__(self,
                 filepath = './',
                 every_n_epoch = -1,
                 every_n_iteration = -1):
        super(ModelCheckpoint).__init__()
        if every_n_epoch != -1:
            assert every_n_iteration == -1
        self.every_n_epoch = every_n_epoch
        if every_n_iteration != -1:
            assert every_n_epoch == -1
        self.every_n_iteration = every_n_iteration

        self.save_path_prefix = filepath
        self.n_iteration = 0

    def on_epoch_end(self, epochs, logs = None):
        if self.every_n_epoch >= 1 and epochs % self.every_n_epoch == 0:
            self._save_model()

    def on_batch_end(self, batch, logs = None):
        if self.every_n_iteration >=1 and self.n_iteration % self.every_n_iteration == 0:
            self._save_model()
        self.n_iteration += 1

    def _save_model(self):
        self.trainercontrol.should_save = True

