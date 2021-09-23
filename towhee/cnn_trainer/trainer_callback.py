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
import dataclasses
import json

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
from towhee.cnn_trainer.training_args import TrainingArguments
from towhee.cnn_trainer.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class TrainerState:
    """
    A class containing the :class:`~PyTorchCNNTrainer` inner state that will be saved along the model and optimizer
    when checkpointing and passed to the :class:`~towhee.TrainerCallback`.
    Args:
        epoch (:obj:`float`, `optional`):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (:obj:`int`, `optional`, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (:obj:`int`, `optional`, defaults to 0):
            The number of update steps to do during the current training.
        total_flos (:obj:`float`, `optional`, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (:obj:`List[Dict[str, float]]`, `optional`):
            The list of logs done since the beginning of training.
        best_metric (:obj:`float`, `optional`):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (:obj:`str`, `optional`):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be :obj:`True` for one process).
        is_hyper_param_search (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
    """

    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside :obj:`json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of :obj:`json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    """
    A class that handles the :class:`~PyTorchCNNTrainer` control flow. This class is used by the
    :class:`~towhee.TrainerCallback` to activate some switches in the training loop.
    Args:
        should_training_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the training should be interrupted.
            If :obj:`True`, this variable will not be set back to :obj:`False`. The training will just stop.
        should_epoch_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the current epoch should be interrupted.
            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next epoch.
        should_save (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be saved at this step.
            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be evaluated at this step.
            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_log (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the logs should be reported at this step.
            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


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

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the :class:`~towhee.Trainer`.
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "Unexpected error."
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                "add_callback error."
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A :class:`~towhee.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        control.should_log = True

        # Save
        control.should_save = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        control.should_log = True

        # Evaluate
        control.should_save = True

        return control


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
