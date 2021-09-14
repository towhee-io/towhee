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
import math
import os
import random
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from towhee.cnn_trainer.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from towhee.cnn_trainer.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
)
from towhee.cnn_trainer.training_args import TrainingArguments
from towhee.cnn_trainer.utils import logging

DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PRO = ProgressCallback

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"


class PyTorchCNNTrainer:
    """
    PyTorchCNNTrainer is a simple but feature-complete training and eval loop for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`):
            The model to train, evaluate or use for predictions.
        args (:class:`~towhee.TrainingArguments`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~towhee.TrainingArguments` with the ``output_dir`` set to a directory named
            `tmp_trainer` in the current directory if not provided.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`):
            The dataset to use for training.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation.
        callbacks (List of :obj:`~towhee.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop.

    """

    def __init__(
            self,
            model: nn.Module = None,
            args: TrainingArguments = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]
            = (None, None),
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info("No `TrainingArguments` passed, using `output_dir.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args

        if model is None:
            raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model

        self.optimizer, self.lr_scheduler = optimizers
        default_callbacks = DEFAULT_CALLBACKS
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)

        os.makedirs(self.args.output_dir, exist_ok=True)

        if args.max_steps > 0:
            logger.info("max_steps is given.")

        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        self.state = TrainerState()
        # control the save condition
        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flcurrent_flosos = 0
        default_label_names = (
            ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def add_callback(self, callback):
        """
        Add a callback to the current list of :class:`~towhee.TrainerCallback`.

        Args:
           callback (:obj:`type` or :class:`~towhee.TrainerCallback`):
               A :class:`~towhee.TrainerCallback` class or an instance of a :class:`~towhee.TrainerCallback`.
               In the first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

    def create_optimizer_and_scheduler(self):
        """
        Setup the optimizer and the learning rate scheduler.
        """
        self.create_optimizer()
        self.create_scheduler()

    def create_optimizer(self):
        """
        Setup the optimizer.
        """
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def create_scheduler(self):
        """
        Setup the scheduler. The optimizer of the cnn_trainer must have been set up before this method is called.
        """
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset does not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def train(self):
        """
        Main training entry point.
        """
        args = self.args

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
        else:
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize

        self.create_optimizer_and_scheduler()

        self.state = TrainerState()

        model = self.model

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0
        epochs_trained = 0

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        tr_loss = torch.tensor(0.0)
        tr_corrects = 0
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        # model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                loss, corrects = self.training_step(model, inputs)
                tr_loss += loss
                tr_corrects += corrects

                # Optimizer step
                optimizer_was_run = True
                self.optimizer.step()

                if optimizer_was_run:
                    self.lr_scheduler.step()

                self.optimizer.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, tr_corrects, num_examples)

            if self.control.should_training_stop:
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item() / num_examples
        # train_loss = self._total_loss_scalar / self.state.global_step
        train_loss = self._total_loss_scalar / num_examples

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss)

    def _maybe_log_save_evaluate(self, tr_loss, tr_corrects, num_examples):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            epoch_loss = tr_loss_scalar / num_examples
            epoch_acc = tr_corrects / num_examples

            # reset tr_loss to zero
            tr_loss -= tr_loss
            tr_corrects -= tr_corrects

            logs["loss"] = epoch_loss
            logs["accuracy"] = epoch_acc

            self._total_loss_scalar += tr_loss_scalar
            self.log(logs)

        self.control.should_save = False
        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)

        # Save optimizer and scheduler
        if self.args.should_save:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True):
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.random.get_rng_state()

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def training_step(self, model: nn.Module, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # inputs = self._prepare_inputs(inputs)

        loss, corrects = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss.backward()

        return loss.detach(), corrects

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by PyTorchCNNTrainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs[1]
        with torch.set_grad_enabled(True):
            outputs = model(inputs[0])
            _, preds = torch.max(outputs, 1)

        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            corrects = torch.sum(preds == labels.data)

        # return (loss, outputs) if return_outputs else loss
        return loss, corrects

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        state_dict = self.model.state_dict()
        if self.args.should_save:
            self._save(output_dir, state_dict)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
