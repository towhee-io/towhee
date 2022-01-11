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

import collections

from tqdm.auto import tqdm
from towhee.trainer.training_config import TrainingConfig
from towhee.trainer.utils import logging

logger = logging.get_logger(__name__)


class TrainerCallback:
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:
    Args:
        args (:class:`~towhee.TrainingArguments`):
            The training arguments used to instantiate the :class:`~towhee.Trainer`.
        state (:class:`~towhee.TrainerState`):
            The current state of the :class:`~towhee.Trainer`.
        control (:class:`~towhee.TrainerControl`):
            The object that is returned to the :class:`~towhee.Trainer` and can be used to make some decisions.
        model (:obj:`torch.nn.Module`):
            The model being trained.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        eval_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        logs  (:obj:`Dict[str, float]`):
            The values to log.
            Those are only accessible in the event :obj:`on_log`.
    The :obj:`control` object is the only one that can be changed by the callback, in which case the event that changes
    it should return the modified version.
    The argument :obj:`args`, :obj:`state` and :obj:`control` are positionals for all events, all the others are
    grouped in :obj:`kwargs`. You can unpack the ones you need in the signature of the event using them. As an example,
    see the code of the simple :class:`~towhee.PrinterCallback`.
    """

    def on_init_end(self ):
        """
        Event called at the end of the initialization of the :class:`~towhee.Trainer`.
        """
        pass

    def on_train_begin(self ):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self ):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self ):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self ):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_step_end(self ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self ):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self ):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self ):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self ):
        """
        Event called after a prediction step.
        """
        pass




class ProgressCallback(TrainerCallback):
    """
    A :class:`~towhee.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and isinstance(eval_dataloader.dataset, collections.abc.Sized):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None)
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare :class:`~towhee.TrainerCallback` that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
