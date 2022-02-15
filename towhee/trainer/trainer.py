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
import tempfile
import torch.distributed as dist

from typing import Union, Dict, Any
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.multiprocessing import Process
from towhee.trainer.metrics import get_metric_by_name
from towhee.trainer.modelcard import ModelCard, MODEL_CARD_NAME
from towhee.trainer.utils.trainer_utils import CHECKPOINT_NAME
from towhee.trainer.training_config import TrainingConfig
from towhee.utils.log import trainer_log
from towhee.trainer.dataset import TowheeDataSet
from towhee.trainer.optimization.optimization import get_scheduler
from towhee.trainer.callback import CallbackList

# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PRO = ProgressCallback

WEIGHTS_NAME = "pytorch_model.bin"
TEMP_INIT_WEIGHTS = "initial_weights.pt"
NAME = "name_"


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
    if world_size < 2:  # 单GPU的情况
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def _construct_loss_from_config(module: Any, config: Union[str, Dict]):
    """
    construct from the config, the config can be class name as a `str`, or a dict containing the construct parameters.
    """
    instance = None
    if isinstance(config, str):
        construct_name = getattr(module, config)
        instance = construct_name()
    elif isinstance(config, Dict):
        optimizer_construct_name = config[NAME]
        construct_name = getattr(module, optimizer_construct_name)
        kwargs = {}
        for arg_name in config:
            if arg_name != NAME:
                kwargs[arg_name] = config[arg_name]
        instance = construct_name(**kwargs)
    return instance


def _construct_optimizer_from_config(module: Any, config: Union[str, Dict], model=None):
    """
    construct from the config, the config can be class name as a `str`, or a dict containing the construct parameters.
    """
    instance = None
    if isinstance(config, str):
        construct_name = getattr(module, config)
        if model is not None:
            instance = construct_name(model.parameters())
    elif isinstance(config, Dict):
        optimizer_construct_name = config[NAME]
        construct_name = getattr(module, optimizer_construct_name)
        kwargs = {}
        for arg_name in config:
            if arg_name != NAME:
                kwargs[arg_name] = config[arg_name]
        if model is not None:
            instance = construct_name(model.parameters(), **kwargs)
    return instance


class Trainer:
    """
    train an operator
    """

    def __init__(
            self,
            model: nn.Module = None,
            training_config: TrainingConfig = None,
            train_dataset: Union[Dataset, TowheeDataSet] = None,
            eval_dataset: Union[Dataset, TowheeDataSet] = None,
            model_card: ModelCard = None
    ):
        if training_config is None:
            output_dir = "tmp_trainer"
            trainer_log.info("No `TrainingArguments` passed, using `output_dir.")
            training_config = TrainingConfig(output_dir=output_dir)
        self.configs = training_config

        if model is None:
            raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")

        if isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset
        elif isinstance(train_dataset, TowheeDataSet):
            self.train_dataset = train_dataset.dataset

        self.eval_dataset = eval_dataset
        self.model = model
        self.model_card = model_card
        self.optimizer = None
        self.lr_scheduler_type = self.configs.lr_scheduler_type
        self.lr_scheduler = None
        self.metric = None
        self.loss = None
        self.loss_val = 0.0
        self.callbacks = CallbackList()
        self.mean_loss_metric = None
        self.epoch = 0

        os.makedirs(self.configs.output_dir, exist_ok=True)
        if training_config.max_steps > 0:
            trainer_log.info("max_steps is given.")
        if train_dataset is not None and not isinstance(train_dataset,
                                                        collections.abc.Sized) and training_config.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        if self.model_card is None:
            self.model_card = ModelCard()
        self.model_card.model_details = str(self.model)
        self.model_card.training_configs = str(self.configs)

    def train(self, resume_checkpoint_path=None):
        if self.configs.device_str == "cuda":
            self.distributed = True
            self.spawn_train_process(resume_checkpoint_path)
        else:
            self.distributed = False
            self.run_train(resume_checkpoint_path)

    def spawn_train_process(self, resume_checkpoint_path):
        # world_size = torch.cuda.device_count()
        # mp.spawn(self.run_train,
        #          args=(world_size, resume_checkpoint_path),
        #          nprocs=world_size,  # opt.world_size,
        #          join=True)
        process_list = []
        world_size = self.configs.n_gpu
        for rank in range(world_size):
            process = Process(target=self.run_train, args=(resume_checkpoint_path, rank, world_size))
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()

    def _init_distributed(self, rank, world_size):
        if self.distributed:
            if torch.cuda.is_available() is False:
                raise EnvironmentError("not find GPU device for training.")
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            print("_init_distributed(), rank=", rank)
            torch.cuda.set_device(rank)
            dist_backend = "nccl"
            dist_url = "env://"
            print("| distributed init (rank {}): {}".format(
                rank, dist_url), flush=True)
            dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                    world_size=world_size, rank=rank)
            dist.barrier()

    def _load_before_train(self, resume_checkpoint_path, rank):
        epochs_trained = 0
        last_loss_val = 0.0
        sync_bn = self.configs.sync_bn
        if resume_checkpoint_path is not None:
            # weights_dict = torch.load(weights_path, map_location=device)
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if model.state_dict()[k].numel() == v.numel()}
            # model.load_state_dict(load_weights_dict, strict=False)
            last_epoch, last_loss = self.load(resume_checkpoint_path)
            epochs_trained = last_epoch + 1
            _, last_loss_val = last_loss
        else:  # if using multi gpu and not resume, must keep model replicas in all processes are the same
            if self.distributed:
                checkpoint_path = os.path.join(tempfile.gettempdir(), TEMP_INIT_WEIGHTS)
                if rank == 0:
                    torch.save(self.model.state_dict(), checkpoint_path)
                dist.barrier()
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.configs.device))
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
            if sync_bn:
                # 使用SyncBatchNorm后训练会更耗时
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.configs.device)
        return epochs_trained, last_loss_val

    def run_train(self, resume_checkpoint_path=None, rank=None, world_size=None):
        """
        Main training entry point.
        """
        args = self.configs
        self._init_distributed(rank, world_size)

        print("device=", self.configs.device)
        print("rank=", rank)
        print("world_size=", world_size)

        self.model = self.model.to(self.configs.device)
        epochs_trained, last_loss_val = self._load_before_train(resume_checkpoint_path, rank)
        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        train_dataloader = self.get_train_dataloader()

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

        self._setup_before_train(num_training_steps=max_steps)

        model = self.model

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        trainer_log.info("***** Running training *****")
        trainer_log.info("  Num examples = %d", num_examples)
        trainer_log.info("  Num Epochs = %d", num_train_epochs)
        trainer_log.info("  Total train batch size  = %d", total_train_batch_size)
        trainer_log.info("  Total optimization steps = %d", max_steps)
        trainer_log.info("****************************")

        # model.zero_grad()

        # tb_writer = None
        if rank == 0 or rank is None:
            trainer_log.warning(args)
        #     print("Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/")
        # tb_writer = SummaryWriter()
        logs = {}
        self.callbacks.on_train_begin(logs)
        global_step = 0
        self.mean_loss_metric = get_metric_by_name("MeanMetric")

        total_loss_val = last_loss_val * epochs_trained
        self.epoch = epochs_trained - 1
        for epoch in range(epochs_trained, num_train_epochs):
            model.train()
            loss_sum = 0.0
            self.callbacks.on_epoch_begin(epochs_trained, logs)
            # steps_in_epoch = (
            #     len(epoch_iterator) if train_dataset_is_sized else args.max_steps
            # )
            if rank == 0 or rank is None:
                train_dataloader = tqdm(train_dataloader, unit="step")  # , file=sys.stdout)
            # for step, inputs in enumerate(train_dataloader):
            for i, inputs in enumerate(train_dataloader):
                self.callbacks.on_train_batch_begin(inputs, logs)
                inputs = [input_.to(self.configs.device) for input_ in inputs]
                labels = inputs[1]
                outputs = model(inputs[0])
                loss, step_metric = self.compute_loss(labels, outputs)
                loss = reduce_value(loss, average=True)
                loss.backward()
                loss = loss.detach()
                loss_sum += loss.item()
                mean_loss = loss_sum / (i + 1)  # update mean losses
                if step_metric is None:
                    show_metric = None
                else:
                    show_metric = round(step_metric.item(), 3)
                if rank == 0 or rank is None:
                    train_dataloader.desc = "[epoch {}/{}] loss={}, metric={}".format(epoch + 1,
                                                                                      int(self.configs.epoch_num),
                                                                                      round(mean_loss, 3),
                                                                                      show_metric)
                # Optimizer step
                optimizer_was_run = True
                self.optimizer.step()
                if optimizer_was_run:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
                # lr = self.lr_scheduler.get_last_lr()[-1]
                # if rank == 0 or rank is None:
                #     tb_writer.add_scalar("loss", loss, global_step)
                #     tb_writer.add_scalar("learning_rate", lr, global_step)
                global_step += 1

                # self.state.global_step += 1
                # self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.callbacks.on_train_batch_end(tuple(inputs), logs)

            # self._maybe_log_save_evaluate(loss_sum, tr_corrects, num_examples)

            total_loss_val += mean_loss
            self.epoch += 1
            self.callbacks.on_epoch_end(epochs_trained, logs)
            # if self.control.should_training_stop:
            #     break

        trainer_log.info("\nTraining completed.\n")
        self.loss_val = round(total_loss_val / (self.epoch + 1), 3)
        training_summary = {
            "device": str(self.configs.device),
            "finetuned_from": resume_checkpoint_path,
            "start_epoch": epochs_trained,
            "num_train_epochs": self.epoch + 1 - epochs_trained,
            "loss": {
                str(self.loss): self.loss_val
            }
        }
        self.model_card.training_summary = training_summary

        self._cleanup_distributed(rank)
        if rank == 0 or rank is None:  # todo
            self.save(
                path=os.path.join(args.output_dir, "epoch_" + str(num_train_epochs)),
                overwrite=args.overwrite_output_dir
            )
        self.callbacks.on_train_end(logs)

    def _cleanup_distributed(self, rank):
        if self.distributed:
            if rank == 0:
                if os.path.exists(TEMP_INIT_WEIGHTS) is True:
                    os.remove(TEMP_INIT_WEIGHTS)
            dist.destroy_process_group()

    def compute_loss(self, labels, outputs):
        """
        Subclass and override for custom behavior.
        """
        #     criterion = nn.CrossEntropyLoss()
        loss = self.loss(outputs, labels)
        # corrects = torch.sum(preds == labels.data)
        # return (loss, outputs) if return_outputs else loss
        step_metric = None
        if self.metric is not None:
            self.metric.update(outputs, labels)
            step_metric = self.metric.compute()
        # self.mean_loss_metric.update(loss)
        # self.mean_loss_metric.compute()
        return loss, step_metric

    def push_model_to_hub(self):
        pass

    def add_callback(self, callback):
        self.callbacks.add_callback(callback)

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

    def _setup_before_train(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        """
        self._create_optimizer()
        self._create_loss()
        self._create_metric()
        self._create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def _create_metric(self):
        self.metric = get_metric_by_name(self.configs.metric).to(self.configs.device)

    def _create_loss(self):
        self.loss = _construct_loss_from_config(torch.nn.modules.loss, self.configs.loss)

    def _create_optimizer(self):
        self.optimizer = _construct_optimizer_from_config(optim, self.configs.optimizer, model=self.model)

    def _create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
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
        checkpoint_path = Path(path).joinpath(CHECKPOINT_NAME)
        modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        checkpoint = torch.load(checkpoint_path, map_location=self.configs.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if isinstance(self.optimizer, Optimizer) and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" not in checkpoint:
            return 0
        epoch = checkpoint["epoch"]
        print("epoch = ", epoch)  # todo
        loss = checkpoint["loss"]
        print("loss = ", loss)  # todo
        if Path(modelcard_path).exists():
            self.model_card = ModelCard.load_from_file(modelcard_path)
            print(f"Model card is loaded from {modelcard_path}")
            # print("model_card = ", self.model_card.to_dict())
        else:
            trainer_log.warning("model card file not exist.")
        return epoch, loss

    def save(self, path, overwrite=True):
        if not overwrite:
            if Path(path).exists():
                raise FileExistsError("File already exists: ", str(Path(path).resolve()))
        Path(path).mkdir(exist_ok=True)
        checkpoint_path = Path(path).joinpath(CHECKPOINT_NAME)
        modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        print("save checkpoint_path:", checkpoint_path)
        optimizer_state_dict = None
        if isinstance(self.optimizer, Optimizer):  # if created
            optimizer_state_dict = self.optimizer.state_dict()
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "loss": (self.loss, self.loss_val),
        }, checkpoint_path)
        if self.model_card is not None:
            self.model_card.save_model_card(modelcard_path)
        else:
            trainer_log.warning("model card is None.")
