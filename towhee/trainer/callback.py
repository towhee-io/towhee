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
from typing import Dict, Tuple, List, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from towhee.utils.log import trainer_log
from towhee.trainer.utils.trainer_utils import is_main_process
__all__ = [
    "Callback",
    "CallbackList",
    "TrainerControl",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
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
                 should_training_stop: bool = False,
                 should_epoch_stop: bool = False,
                 should_save:bool = False,
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
        """
        Set the model to callback.

        Args:
            model (`torch.nn.Module`):
                The model which callback can operate.
        """
        self.model = model

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        Set the optimizer to callback.

        Args:
            optimizer (`torch.optim.Optimizer`):
                The optimizer which callback can operate.
        """
        self.optimizer = optimizer

    def set_trainercontrol(self, trainercontrol: TrainerControl) -> None:
        """
        Set the trainercontrol to callback.

        Args:
            trainercontrol (`towhee.trainer.callback.TrainerControl`):
                The trainercontrol which callback can operate.
        """
        self.trainercontrol = trainercontrol

    def on_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before every batch calculation.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        pass

    def on_batch_end(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked after every batch calculation.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        pass

    def on_epoch_begin(self, epochs: int, logs: Dict) -> None:
        """
        Hook function invoked before each epoch.

        Args:
            epochs (`int`):
                Epoch index.
            logs (`Dict`):
                Kv store to save and load info.
        """
        pass

    def on_epoch_end(self, epochs: int, logs: Dict) -> None:
        """
        Hook function invoked after each epoch.

        Args:
            epochs (`int`):
                Epoch index.
            logs (`Dict`):
                Kv store to save and load info.
        """
        pass

    def on_train_begin(self, logs: Dict) -> None:
        """
        Hook function invoked before train stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        pass

    def on_train_end(self, logs: Dict) -> None:
        """
        Hook function invoked after train stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        pass

    def on_train_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before train stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        self.on_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before every batch calculation in train stage.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        self.on_batch_end(batch, logs)

    def on_eval_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before every batch calculation in evaluate stage.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        self.on_batch_begin(batch, logs)

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked after every batch calculation in evaluate stage.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        self.on_batch_end(batch, logs)

    def on_eval_begin(self, logs: Dict) -> None:
        """
        Hook function invoked before evaluate stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        pass

    def on_eval_end(self, logs: Dict) -> None:
        """
        Hook function invoked after evaluate stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        pass


class CallbackList:
    """
    `CallbackList` aggregate multiple `Callback` in the same object. Invoke the
    callbacks of `CallbackList` will invoke corresponding callback in each
    "Callback" in the FIFO sequential order.

    Args:
        callbacks (`List[towhee.trainer.callback.Callback]`)
            A list of callbacks which methods will be called simultaneously.

    Example:

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
        """
        Set the model to callback.

        Args:
            model (`torch.nn.Module`):
                The model which callback can operate.
        """
        self.model = model
        for cb in self.callbacks:
            cb.set_model(model)

    def set_optimizer(self, optimizer: Optimizer):
        """
        Set the optimizer to callback.

        Args:
            optimizer (`torch.optim.Optimizer`):
                The optimizer which callback can operate.
        """
        self.optimizer = optimizer
        for cb in self.callbacks:
            cb.set_optimizer(optimizer)

    def set_trainercontrol(self, trainercontrol: TrainerControl):
        """
        Set the trainercontrol to callback.

        Args:
            trainercontrol (`towhee.trainer.callback.TrainerControl`):
                The trainercontrol which callback can operate.
        """
        self.trainercontrol = TrainerControl
        for cb in self.callbacks:
            cb.set_trainercontrol(trainercontrol)

    def add_callback(self, callback: Callback, singleton: bool = True):
        """
        Args:
            callback (`towhee.trainer.callback.Callback`):
                The callback need to be added.
            singleton (`bool`):
                If set true, only one instance of same `Callback` will remain in callbacklist.
        """
        if singleton:
            for old_callback in self.callbacks:
                if old_callback.__class__.__name__ == callback.__class__.__name__:
                    self.callbacks.remove(old_callback)
        self.callbacks.append(callback)

    def pop_callback(self, callback: Callback):
        """
        Args:
            callback (`towhee.trainer.callback.Callback`)
                The callback need to be removed from callback list.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def on_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before every batch calculation.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked after every batch calculation.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)

    def on_epoch_begin(self, epochs: int, logs: Dict) -> None:
        """
        Hook function invoked before each epoch.

        Args:
            epochs (`int`):
                Epoch index.
            logs (`Dict`):
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_epoch_begin(epochs, logs)

    def on_epoch_end(self, epochs: int, logs: Dict) -> None:
        """
        Hook function invoked after each epoch.

        Args:
            epochs (`int`):
                Epoch index.
            logs (`Dict`):
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_epoch_end(epochs, logs)

    def on_train_begin(self, logs: Dict) -> None:
        """
        Hook function invoked before train stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Dict) -> None:
        """
        Hook function invoked after train stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_train_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before train stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before every batch calculation in train stage.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs)

    def on_eval_batch_begin(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked before every batch calculation in evaluate stage.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_eval_batch_begin(batch, logs)

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        """
        Hook function invoked after every batch calculation in evaluate stage.

        Args:
            batch (`Tuple`):
                The data batch to calculate.
            logs (`Dict`)
                Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_eval_batch_end(batch, logs)

    def on_eval_begin(self, logs: Dict) -> None:
        """
        Hook function invoked before evaluate stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_eval_begin(logs)

    def on_eval_end(self, logs: Dict) -> None:
        """
        Hook function invoked after evaluate stage.

        Args:
            logs (`Dict`):
               Kv store to save and load info.
        """
        for cb in self.callbacks:
            cb.on_eval_end(logs)


class EarlyStoppingCallback(Callback):
    """
    EarlyStoppingCallback

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`.
    Training loop will check at end of every epoch whether the loss is no
    longer decreasing, considering the `min_delta` and `patience` if
    applicable. Once it's found no longer decreasing. `trainercontrol.
    should_training_stop` is marked True.

    Args:
        trainercontrol (`towhee.trainer.callback.TrainerControl`):
            The trainercontrol which callback can operate.
        monitor (`str`):
            Quantity to be monitored.
        min_delta (`float`):
            Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta,
            will count as no improvement.
        patience (`str`):
            Number of epochs with no improvement after which training will
            be stopped.
        mode (`str`):
            One of `{"min", "max"}`. In `min` mode, training will stop when
            the quantity monitored has stopped decreasing; in `"max"`
            mode it will stop when the quantity monitored has stopped increasing.
        baseline (`float`):
            Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
    """

    def __init__(self,
                 trainercontrol: TrainerControl,
                 monitor: str,
                 min_delta: float = 0,
                 patience: int = 0,
                 mode: str = "max",
                 baseline: float = None
                 ):
        super(EarlyStoppingCallback).__init__()
        self.trainercontrol = trainercontrol
        self.monitor = monitor
        self.patience = patience
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

    def on_train_begin(self, logs: Dict = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epochs: int, logs: Dict = None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epochs
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience and epochs > 0 and current != 0:
            self.stopped_epoch = epochs
            self.trainercontrol.should_training_stop = True

    def on_train_end(self, logs: Dict = None):
        if self.stopped_epoch > 0:
            if is_main_process():
                trainer_log.warning(
                    "monitoring %s not be better then %s on epoch %s for waiting for %s epochs. Early stop on epoch %s.",
                    self.monitor, self.best, self.best_epoch, self.wait, self.stopped_epoch)

    def get_monitor_value(self, logs: Dict):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)

        return monitor_value

    def _is_improvement(self, monitor_value: float, reference_value: float):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class ModelCheckpointCallback(Callback):
    """
    ModelCheckpointCallback is intended to save the model at some interval. It can
    be set in epoch mode or iteration mode. Only one of `every_n_epoch` and
    `every_n_iteration` can be set to a positive value and the `trainer.should_save`
    will set to True when the condion meets.

    Args:
        trainercontrol (`TrainerControl`):
            The trainercontrol which callback can operate.
        filepath (`str`):
            Filepath to save the model.
        every_n_epoch (`int`):
            Save the model after n epochs.
        every_n_iteration (`int`):
            Save the model after n iterations.
    """

    def __init__(self,
                 trainercontrol: TrainerControl,
                 filepath: str = "./",
                 every_n_epoch: int = -1,
                 every_n_iteration: int = -1):
        super(ModelCheckpointCallback).__init__()
        self.trainercontrol = trainercontrol
        if every_n_epoch != -1:
            assert every_n_iteration == -1
        self.every_n_epoch = every_n_epoch
        if every_n_iteration != -1:
            assert every_n_epoch == -1
        self.every_n_iteration = every_n_iteration

        self.save_path_prefix = filepath
        self.n_iteration = 0
        assert(self.every_n_epoch != 0 and self.every_n_epoch > -2)
        assert(self.every_n_iteration != 0 and self.every_n_iteration > -2)

    def on_epoch_end(self, epochs: int, logs: Dict = None):
        if self.every_n_epoch == -1:
            return
        if self.trainercontrol.should_save is True:
            self.trainercontrol.should_save = False
        if self.every_n_epoch >= 1 and epochs % self.every_n_epoch == 0:
            self._save_model()

    def on_batch_end(self, batch: Tuple, logs: Dict = None):
        if self.every_n_iteration == -1:
            return
        if self.trainercontrol.should_save is True:
            self.trainercontrol.should_save = False
        if self.every_n_iteration >= 1 and self.n_iteration % self.every_n_iteration == 0:
            self._save_model()
        self.n_iteration += 1

    def _save_model(self):
        self.trainercontrol.should_save = True

class TensorBoardCallBack(Callback):
    """
    TensorBoardCallBack is intended to record the essential value(e.g. epoch_loss)
    to tensorboard after each iteration. If tensorboard is available, you can see
    the tensorboard in localhost:6006.

    Args:
        summary_writer_constructor (`Callable`):
            Function which construct tensorboard summary writer.
        log_dir (`str`):
            Save directory location.
        comment (`str`):
            Comment log_dir suffix appended to the default log_dir.
    """

    def __init__(self, summary_writer_constructor: Callable, log_dir: str = None, comment:str = ""):
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
    PrintCallBack is intended to print logs on the screen.

    Args:
        total_epoch_num (`int`):
            Epoch numbers expected to run.
        step_frequency (`int`):
            Print information in every n steps.
    """

    def __init__(self, total_epoch_num: int, step_frequency:int = 16):
        super().__init__()
        self.step_frequency = step_frequency
        self.total_epoch_num = total_epoch_num

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        if is_main_process():
            global_step = logs["global_step"]
            if global_step % self.step_frequency == 0:
                print("epoch={}/{}, global_step={}, epoch_loss={}, epoch_metric={}"
                      .format(logs["epoch"], self.total_epoch_num,
                              global_step,
                              logs["epoch_loss"],
                              logs["epoch_metric"]))

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        if is_main_process():
            eval_global_step = logs["eval_global_step"]
            if eval_global_step % self.step_frequency == 0:
                print("epoch={}/{}, eval_global_step={}, eval_epoch_loss={}, eval_epoch_metric={}"
                      .format(logs["epoch"], self.total_epoch_num,
                              eval_global_step,
                              logs["eval_epoch_loss"],
                              logs["eval_epoch_metric"]))


class ProgressBarCallBack(Callback):
    """
    ProgressBarCallBack is intended to print a progress bar to visualize current
    training progress. The tqdm is used as the progress bar backend.

    Args:
        total_epoch_num (`int`):
            Epoch numbers expected to run.
        train_dataloader (`torch.utils.data.DataLoader`):
            training dataloader for tqdm to warp.
    """

    def __init__(self, total_epoch_num: int, train_dataloader: torch.utils.data.DataLoader):
        super().__init__()
        self.total_epoch_num = total_epoch_num
        self.raw_train_dataloader = train_dataloader
        self.now_tqdm_train_dataloader: tqdm = train_dataloader
        self.description = ""

    def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
        if is_main_process():
            self.now_tqdm_train_dataloader.update(1)
            self.description = "[epoch {}/{}] loss={}, metric={}".format(logs["epoch"],
                                                                         int(self.total_epoch_num),
                                                                         round(logs["epoch_loss"], 3),
                                                                         round(logs["epoch_metric"], 3))
            self.now_tqdm_train_dataloader.set_description(self.description)

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
            self.now_tqdm_train_dataloader.set_description(self.description)
