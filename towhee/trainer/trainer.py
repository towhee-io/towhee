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
import sys
import torch
# import random
# import warnings
# import numpy as np

from typing import Dict, List, Union
from pathlib import Path

from torch import nn
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

# from towhee.trainer.callback import (
#     Callback
#     CallbackHandler,
#     DefaultFlowCallback,
#     PrinterCallback,
#     ProgressCallback,
#     Callback,
#     TrainerControl,
#     TrainerState,
# )
from towhee.trainer.modelcard import ModelCard, MODEL_CARD_NAME

from towhee.trainer.utils.trainer_utils import (
    CHECKPOINT_NAME,
    get_last_checkpoint
)
from towhee.trainer.training_config import TrainingConfig
from towhee.trainer.utils import logging
from towhee.trainer.dataset import TowheeDataSet
from towhee.trainer.optimization.optimization import get_scheduler

# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PRO = ProgressCallback

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"


class Trainer:
    """
    train an operator
    """

    def __init__(
            self,
            # operator: "NNOperator" = None,
            model: nn.Module = None,
            training_config: TrainingConfig = None,
            train_dataset: Union[Dataset, TowheeDataSet] = None,
            eval_dataset: Union[Dataset, TowheeDataSet] = None,
            model_card: ModelCard = None
            # callbacks: Optional[List[Callback]] = None,
            # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]
            # = (None, None),
    ):
        if training_config is None:
            output_dir = "tmp_trainer"
            logger.info("No `TrainingArguments` passed, using `output_dir.")
            training_config = TrainingConfig(output_dir=output_dir)
        self.configs = training_config
        # operator.model
        # model = operator.get_model()
        if model is None:
            raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")

        if isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset
        elif isinstance(train_dataset, TowheeDataSet):
            self.train_dataset = train_dataset.dataset

        self.eval_dataset = eval_dataset
        # self.operator = operator
        self.model = model
        self.model_card = model_card
        self.checkpoint = None
        self.optimizer = self.configs.optimizer
        self.lr_scheduler_type = self.configs.lr_scheduler_type
        self.lr_scheduler = None
        # default_callbacks = DEFAULT_CALLBACKS
        # callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        # self.callback_handler = CallbackHandler(
        #     callbacks, self.model, self.optimizer, self.lr_scheduler
        # )

        # self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)

        os.makedirs(self.configs.output_dir, exist_ok=True)

        if training_config.max_steps > 0:
            logger.info("max_steps is given.")

        if train_dataset is not None and not isinstance(train_dataset,
                                                        collections.abc.Sized) and training_config.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        # self.state = TrainerState()
        # control the save condition
        # self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flcurrent_flosos = 0
        default_label_names = (
            ["labels"]
        )
        self.label_names = default_label_names if self.configs.label_names is None else self.configs.label_names
        # self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def train(self, epochs_trained: int = 0):
        """
        Main training entry point.
        """
        args = self.configs

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
                max_steps = math.ceil(args.epoch_num * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.epoch_num)
        else:
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()

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

        # self.state.epoch = 0

        # Update the references
        # self.callback_handler.model = self.model
        # self.callback_handler.optimizer = self.optimizer
        # self.callback_handler.lr_scheduler = self.lr_scheduler
        # self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        # self.state.max_steps = max_steps
        # self.state.num_train_epochs = num_train_epochs

        tr_loss = torch.tensor(0.0)
        tr_corrects = 0
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        # model.zero_grad()

        # self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for _ in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # steps_in_epoch = (
            #     len(epoch_iterator) if train_dataset_is_sized else args.max_steps
            # )
            # self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for _, inputs in enumerate(epoch_iterator):
                loss, corrects = self.training_step(model, inputs)
                print(loss)
                tr_loss += loss
                tr_corrects += corrects

                # Optimizer step
                optimizer_was_run = True
                self.optimizer.step()

                if optimizer_was_run:
                    self.lr_scheduler.step()

                self.optimizer.zero_grad()
                # self.state.global_step += 1
                # self.state.epoch = epoch + (step + 1) / steps_in_epoch
                # self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            # self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self._maybe_log_save_evaluate(tr_loss, tr_corrects, num_examples)

            # if self.control.should_training_stop:
            #     break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item() / num_examples
        # train_loss = self._total_loss_scalar / self.state.global_step
        # train_loss = self._total_loss_scalar / num_examples

        # self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # return TrainOutput(self.state.global_step, train_loss)

        self.save(
            path=os.path.join(args.output_dir, "epoch_" + str(num_train_epochs)),
            overwrite=args.overwrite_output_dir
        )

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
        self.control = self.callback_handler.on_log(self.configs, self.state, self.control, logs)

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

        if self.configs.n_gpu > 1:
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

    def push_model_to_hub(self):
        pass

    def add_callback(self, callback):
        """
        Add a callback to the current list of :class:`~towhee.TrainerCallback`.

        Args:
           callback (:obj:`type` or :class:`~towhee.TrainerCallback`):
               A :class:`~towhee.TrainerCallback` class or an instance of a :class:`~towhee.TrainerCallback`.
               In the first case, will instantiate a member of that class.
        """
        # self.callback_handler.add_callback(callback)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, IterableDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.configs.train_batch_size,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.configs.train_batch_size,
            shuffle=True
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_optimizer(self):
        """
        Setup the optimizer.
        """
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = self.configs.optimizer(self.model.parameters())

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.configs.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.configs.warmup_steps if self.configs.warmup_steps > 0 else math.ceil(
                num_training_steps * self.configs.warmup_ratio)
        )
        return warmup_steps

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset does not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def load(self, path):
        device = "cpu"  # todo
        checkpoint_path = Path(path).joinpath(CHECKPOINT_NAME)
        modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        if isinstance(self.optimizer, Optimizer) and self.checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        epoch = self.checkpoint["epoch"]
        print("epoch = ", epoch)  # todo
        loss = self.checkpoint["loss"]
        print("loss = ", loss)  # todo
        if Path(modelcard_path).exists():
            self.model_card = ModelCard.load_from_file(modelcard_path)
            print("model_card = ", self.model_card.to_dict())
        else:
            logger.warning("model card file not exist.")

    def save(self, path, overwrite=True):
        Path(path).mkdir(exist_ok=True)
        if not overwrite:
            if Path(path).exists():
                raise FileExistsError("File already exists: ", str(Path(path).resolve()))
        checkpoint_path = Path(path).joinpath(CHECKPOINT_NAME)
        modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        print("save checkpoint_path:", checkpoint_path)
        optimizer_state_dict = None
        if isinstance(self.optimizer, Optimizer):  # if created
            optimizer_state_dict = self.optimizer.state_dict()
        torch.save({
            "epoch": 20,  # todo
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "loss": 0.12,  # todo
        }, checkpoint_path)
        if self.model_card is not None:
            self.model_card.save_model_card(modelcard_path)
        else:
            logger.warning("model card is None.")

    def resume_from_checkpoint(self, last_checkpoint=None):
        if last_checkpoint is None:
            last_checkpoint = get_last_checkpoint(self.configs.output_dir)
        self.load(last_checkpoint)
        self.train(epochs_trained=self.checkpoint["epoch"])

# if __name__ == '__main__':
#     from torchvision import transforms
#     # data_train = datasets.MNIST(root="./data/",
#     #                             transform=transform,
#     #                             train=True,
#     #                             download=True)
#     #
#     # data_test = datasets.MNIST(root="./data/",
#     #                            transform=transform,
#     #                            train=False)
#     # data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
#     #                                                 batch_size=64,
#     #                                                 shuffle=True)
#     #
#     # data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
#     #                                                batch_size=64,
#     #                                                shuffle=True)
#
#     class Model(torch.nn.Module):
#         def __init__(self):
#             super(Model, self).__init__()
#             self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#                                              torch.nn.ReLU(),
#                                              torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#                                              torch.nn.ReLU(),
#                                              torch.nn.MaxPool2d(stride=2, kernel_size=2))
#             self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
#                                              torch.nn.ReLU(),
#                                              torch.nn.Dropout(p=0.5),
#                                              torch.nn.Linear(1024, 10))
#
#         def forward(self, x):
#             x = self.conv1(x)
#             x = x.view(-1, 14 * 14 * 128)
#             x = self.dense(x)
#             return x
#
#     model = Model()
#     cost = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())
#
#     training_config = TrainingConfig(
#             output_dir='./temp_output',
#             overwrite_output_dir=True,
#             epoch_num=2,
#             per_gpu_train_batch_size=64,
#             prediction_loss_only=True,
#         )
#     mnist_transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.5], std=[0.5])])
#     data_train = get_dataset('mnist', transform=mnist_transform)
#     # data_train = get_dataset('imdb', root='./data1', split='test')#, transform=mnist_transform)
#     # s = data_train.dataset.extra_repr()
#     # print('1')
#     # print(isinstance(data_train, torch.utils.data.IterableDataset))
#     # print(s)
#     print(data_train.get_framework())
#     trainer = Trainer(
#         model=model,
#         training_config=training_config,
#         train_dataset=data_train
#     )
#     trainer.train()
