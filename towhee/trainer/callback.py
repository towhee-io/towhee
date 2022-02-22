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
import importlib
import numpy as np

from typing import Dict, Tuple, List
from towhee.utils.log import trainer_log
from towhee.trainer.utils.trainer_utils import is_main_process
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer

__all__ = [
    "Callback",
    "CallbackList",
    "TrainerControl",
    "EarlyStopping",
    "ModelCheckpoint",
    "TensorBoardCallBack",
    "PrintCallBack",
    "ProgressBarCallBack"
]

def _get_summary_writer_constructor():
    try:
        tensorboard_module = importlib.import_module("torch.utils.tensorboard")
        summary_writer_constructor = tensorboard_module.SummaryWriter
        trainer_log.info("Use tensorboard. And please observe the logs in  http://localhost:6007/")
        return summary_writer_constructor
    except ImportError:
        trainer_log.info("can not import tensorboard.")
        return None

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
                 should_training_stop=False,
                 should_epoch_stop=False,
                 should_save=False,
                 should_evaluate=False,
                 should_log=False):
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
    "Callback" in the FIFO sequential order.
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
        callback_desc = ""
        for cb in self.callbacks:
            callback_desc += cb.__repr__() + ","
        return "towhee.trainer.CallbackList([{}])".format(callback_desc)

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
                 trainercontrol,
                 monitor,
                 min_delta=0,
                 patience=0,
                 mode="max",
                 baseline=None
                 ):
        # super(EarlyStopping, self).__init__()
        super(EarlyStopping).__init__()
        self.trainercontrol = trainercontrol
        self.monitor = monitor
        self.patience = patience
        # self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        assert mode in ["max", "min"]

        if mode == "min":
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epochs, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        # if self.restore_best_weights and self.best_weights is None:
        #    self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epochs
            # if self.restore_best_weights:
            #     self.best_weights = self.model.get_weights()
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience and epochs > 0 and current != 0:
            self.stopped_epoch = epochs
            self.trainercontrol.should_training_stop = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if is_main_process():
                trainer_log.warning(
                    "monitoring %s not be better then %s on epoch %s for waiting for %s epochs. Early stop on epoch %s.",
                    self.monitor, self.best, self.best_epoch, self.wait, self.stopped_epoch)

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
                 trainercontrol,
                 filepath="./",
                 every_n_epoch=-1,
                 every_n_iteration=-1):
        super(ModelCheckpoint).__init__()
        self.trainercontrol = trainercontrol
        if every_n_epoch != -1:
            assert every_n_iteration == -1
        self.every_n_epoch = every_n_epoch
        if every_n_iteration != -1:
            assert every_n_epoch == -1
        self.every_n_iteration = every_n_iteration

        self.save_path_prefix = filepath
        self.n_iteration = 0

    def on_epoch_end(self, epochs, logs=None):
        if self.every_n_epoch >= 1 and epochs % self.every_n_epoch == 0:
            self._save_model()

    def on_batch_end(self, batch, logs=None):
        if self.every_n_iteration >= 1 and self.n_iteration % self.every_n_iteration == 0:
            self._save_model()
        self.n_iteration += 1

    def _save_model(self):
        self.trainercontrol.should_save = True


class TensorBoardCallBack(Callback):
    """
    if tensorboard is available, you can see the tensorboard in localhost:6006
    """

    def __init__(self, summary_writer_constructor, log_dir=None, comment=""):
        super().__init__()
        self.tb_writer = summary_writer_constructor(log_dir, comment=comment)

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        global_step = logs["global_step"]
        step_loss = logs["step_loss"]
        epoch_loss = logs["epoch_loss"]
        epoch_metric = logs["epoch_metric"]
        lr = logs["lr"]
        if is_main_process():
            self.tb_writer.add_scalar("lr", lr, global_step)
            self.tb_writer.add_scalar("epoch_loss", epoch_loss, global_step)
            self.tb_writer.add_scalar("step_loss", step_loss, global_step)
            self.tb_writer.add_scalar("epoch_metric", epoch_metric, global_step)

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        eval_global_step = logs["eval_global_step"]
        eval_step_loss = logs["eval_step_loss"]
        eval_epoch_loss = logs["eval_epoch_loss"]
        eval_epoch_metric = logs["eval_epoch_metric"]
        if is_main_process():
            self.tb_writer.add_scalar("eval_step_loss", eval_step_loss, eval_global_step)
            self.tb_writer.add_scalar("eval_epoch_loss", eval_epoch_loss, eval_global_step)
            self.tb_writer.add_scalar("eval_epoch_metric", eval_epoch_metric, eval_global_step)


class PrintCallBack(Callback):
    """
    print logs on the screen
    """

    def __init__(self, total_epoch_num, step_frequency=16):
        super().__init__()
        self.step_frequency = step_frequency
        self.total_epoch_num = total_epoch_num

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        global_step = logs["global_step"]
        if global_step % self.step_frequency == 0:
            print("epoch={}/{}, global_step={}, epoch_loss={}, epoch_metric={}"
                  .format(logs["epoch"], self.total_epoch_num,
                          global_step,
                          logs["epoch_loss"],
                          logs["epoch_metric"]))

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        eval_global_step = logs["eval_global_step"]
        if eval_global_step % self.step_frequency == 0:
            print("epoch={}/{}, eval_global_step={}, eval_epoch_loss={}, eval_epoch_metric={}"
                  .format(logs["epoch"], self.total_epoch_num,
                          eval_global_step,
                          logs["eval_epoch_loss"],
                          logs["eval_epoch_metric"]))


class ProgressBarCallBack(Callback):
    """
    use tqdm as the progress bar backend
    """

    def __init__(self, total_epoch_num, train_dataloader):
        super().__init__()
        self.total_epoch_num = total_epoch_num
        self.raw_train_dataloader = train_dataloader
        self.now_tqdm_train_dataloader: tqdm = train_dataloader
        self.descrpition = ""

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        if is_main_process():
            self.now_tqdm_train_dataloader.update(1)
            self.descrpition = "[epoch {}/{}] loss={}, metric={}".format(logs["epoch"],
                                                                         int(self.total_epoch_num),
                                                                         round(logs["epoch_loss"], 3),
                                                                         round(logs["epoch_metric"], 3))
            self.now_tqdm_train_dataloader.set_description(self.descrpition)

    def on_epoch_begin(self, epochs: int, logs: Dict) -> None:
        if is_main_process():
            self.now_tqdm_train_dataloader = None
            self.now_tqdm_train_dataloader = tqdm(self.raw_train_dataloader,
                                                  total=len(self.raw_train_dataloader),
                                                  unit="step")  # , file=sys.stdout)

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        if is_main_process():
            self.description = "[epoch {}/{}] loss={}, metric={}, eval_loss={}, eval_metric={}".format(
                logs["epoch"],
                int(self.total_epoch_num),
                round(logs["epoch_loss"], 3),
                round(logs["epoch_metric"], 3), round(logs["eval_epoch_loss"], 3), round(logs["eval_epoch_metric"], 3))
            self.now_tqdm_train_dataloader.set_description(self.descrpition)
