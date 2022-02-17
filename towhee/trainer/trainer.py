# coding=utf-8
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
import collections
import importlib
import math
import os
import sys
import torch
import tempfile
import torch.distributed as dist

from typing import Union, Dict, Any, Tuple, Optional
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.multiprocessing import Process
# from towhee.trainer.metrics import get_metric_by_name
from towhee.trainer.modelcard import ModelCard, MODEL_CARD_NAME
from towhee.trainer.utils.trainer_utils import CHECKPOINT_NAME, set_seed
from towhee.trainer.utils.trainer_utils import EvalStrategyType
from towhee.trainer.training_config import TrainingConfig
from towhee.utils.log import trainer_log
from towhee.trainer.dataset import TowheeDataSet, TorchDataSet
from towhee.trainer.optimization.optimization import get_scheduler
from towhee.trainer.callback import CallbackList, Callback

# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PRO = ProgressCallback

WEIGHTS_NAME = "pytorch_model.bin"
TEMP_INIT_WEIGHTS = "initial_weights.pt"
NAME = "name_"
CUSTOM = "custom_"


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
    if world_size < 2:  # one gpu
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


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
                  .format(logs["epoch"] + 1, self.total_epoch_num,
                          global_step,
                          logs["epoch_loss"],
                          logs["epoch_metric"]))

    def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
        eval_global_step = logs["eval_global_step"]
        if eval_global_step % self.step_frequency == 0:
            print("epoch={}/{}, eval_global_step={}, eval_epoch_loss={}, eval_epoch_metric={}"
                  .format(logs["epoch"] + 1, self.total_epoch_num,
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
            self.descrpition = "[epoch {}/{}] loss={}, metric={}".format(logs["epoch"] + 1,
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
            self.descrpition = "[epoch {}/{}] loss={}, metric={}, eval_loss={}, eval_metric={}".format(
                logs["epoch"] + 1,
                int(self.total_epoch_num),
                round(logs["epoch_loss"], 3),
                round(logs["epoch_metric"], 3), round(logs["eval_epoch_loss"], 3), round(logs["eval_epoch_metric"], 3))
            self.now_tqdm_train_dataloader.set_description(self.descrpition)


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


def get_summary_writer_constructor():
    try:
        tensorboard_module = importlib.import_module("torch.utils.tensorboard")
        summary_writer_constructor = tensorboard_module.SummaryWriter
        trainer_log.info("Use tensorboard. And please observe the logs in  http://localhost:6007/")
        return summary_writer_constructor
    except ImportError:
        trainer_log.info("can not import tensorboard.")
        return None


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
        self.override_optimizer = False
        self.lr_scheduler_type = self.configs.lr_scheduler_type
        self.lr_scheduler = None
        self.metric = None
        self.loss = None
        self.override_loss = False
        self.loss_value = 0.0
        self.callbacks = CallbackList()
        self.loss_metric = None
        self.metric_value = 0.0
        self.epoch = 0

        os.makedirs(self.configs.output_dir, exist_ok=True)
        if training_config.max_steps > 0:
            trainer_log.info("max_steps is given.")
        if train_dataset is not None and not isinstance(train_dataset,
                                                        collections.abc.Sized) and training_config.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        if not isinstance(self.model_card, ModelCard):
            self.model_card = ModelCard()

        if self.model_card.model_name is not None:
            self.model_card.model_name = type(self.model).__name__
        self.model_card.model_architecture = str(self.model)
        self.model_card.training_config = self.configs


    def train(self, resume_checkpoint_path=None):
        if self.configs.device_str == "cuda":
            self.distributed = True
            self._spawn_train_process(resume_checkpoint_path)
        else:
            self.distributed = False
            self.run_train(resume_checkpoint_path)

    def _spawn_train_process(self, resume_checkpoint_path):
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
        last_loss_value = 0.0
        last_metric_value = 0.0
        sync_bn = self.configs.sync_bn
        if resume_checkpoint_path is not None:
            # weights_dict = torch.load(weights_path, map_location=device)
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if model.state_dict()[k].numel() == v.numel()}
            # model.load_state_dict(load_weights_dict, strict=False)
            last_epoch, last_loss, last_metric = self.load(resume_checkpoint_path)
            epochs_trained = last_epoch + 1
            _, last_loss_value = last_loss
            _, last_metric_value = last_metric
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
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.configs.device)
        return epochs_trained, last_loss_value, last_metric_value

    def _create_logs(self):
        logs = {"global_step": 0, "epoch": self.epoch + 1}
        if self.configs.eval_strategy != EvalStrategyType.NO:
            logs["eval_global_step"] = 0
        return logs

    def run_train(self, resume_checkpoint_path=None, rank=None, world_size=None):
        """
        Main training entry point.
        """
        args = self.configs
        set_seed(self.configs.seed)
        self._init_distributed(rank, world_size)

        print("device=", self.configs.device)
        print("rank=", rank)
        print("world_size=", world_size)

        self.model = self.model.to(self.configs.device)
        epochs_trained, last_loss_value, last_metric_value = self._load_before_train(resume_checkpoint_path, rank)
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

        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        trainer_log.info("***** Running training *****")
        trainer_log.info("  Num examples = %d", num_examples)
        trainer_log.info("  Num Epochs = %d", num_train_epochs)
        trainer_log.info("  Total train batch size  = %d", total_train_batch_size)
        trainer_log.info("  Total optimization steps = %d", max_steps)
        trainer_log.info("****************************")

        if rank == 0 or rank is None:
            trainer_log.warning(args)

        total_metric_value = last_metric_value * epochs_trained
        total_loss_value = last_loss_value * epochs_trained
        self.epoch = epochs_trained - 1
        logs = self._create_logs()
        self.callbacks.on_train_begin(logs)
        for _ in range(epochs_trained, num_train_epochs):
            model.train()
            # batch_loss_sum = 0.0
            self.callbacks.on_epoch_begin(epochs_trained, logs)
            self.loss_metric.reset()
            self.metric.reset()
            # steps_in_epoch = (
            #     len(epoch_iterator) if train_dataset_is_sized else args.max_steps
            # )
            # if rank == 0 or rank is None:
            #     train_dataloader = tqdm(train_dataloader, unit="step")  # , file=sys.stdout)
            # for step, inputs in enumerate(train_dataloader):
            for _, inputs in enumerate(train_dataloader):
                self.callbacks.on_train_batch_begin(inputs, logs)
                inputs = [input_.to(self.configs.device) for input_ in inputs]
                step_logs = self.train_step(model, inputs)  # , train_dataloader)
                logs["lr"] = self.lr_scheduler.get_lr()[0]
                logs["global_step"] += 1
                logs.update(step_logs)
                # self.state.global_step += 1
                # self.state.epoch = epoch + (step + 1) / steps_in_epoch
                if self.configs.eval_strategy == EvalStrategyType.STEP:
                    eval_logs = self.evaluate(model, logs)
                    logs.update(eval_logs)
                self.callbacks.on_train_batch_end(tuple(inputs), logs)
            # self._maybe_log_save_evaluate(loss_sum, tr_corrects, num_examples)
            total_loss_value += logs["epoch_loss"]
            total_metric_value += logs["epoch_metric"]
            if self.configs.eval_strategy == EvalStrategyType.EPOCH:
                eval_logs = self.evaluate(model, logs)
                logs.update(eval_logs)
            self.epoch += 1
            logs["epoch"] = self.epoch + 1
            self.callbacks.on_epoch_end(epochs_trained, logs)
            # if self.control.should_training_stop:
            #     break
        trainer_log.info("\nTraining completed.\n")
        self.loss_value = round(total_loss_value / (self.epoch + 1), 3)
        self.metric_value = round(total_metric_value / (self.epoch + 1), 3)
        training_summary = {
            "device": str(self.configs.device),
            "finetuned_from": resume_checkpoint_path,
            "start_epoch": epochs_trained,
            "num_train_epochs": self.epoch + 1 - epochs_trained,
            "loss": {
                "loss type": str(self.loss),
                "loss value": self.loss_value
            },
            "metric": {
                "metric type": str(self.metric), #todo
                "metric value": self.metric_value
            }
        }

        self.model_card.training_summary = training_summary

        self._cleanup_distributed(rank)
        if is_main_process():
            self.save(
                path=os.path.join(args.output_dir, "epoch_" + str(num_train_epochs)),
                overwrite=args.overwrite_output_dir
            )
        self.callbacks.on_train_end(logs)

    def evaluate_step(self, model, inputs):
        labels = inputs[1]
        outputs = model(inputs[0])
        step_loss = self.compute_loss(labels, outputs)
        step_loss = reduce_value(step_loss, average=True)
        step_loss = step_loss.detach()

        loss_metric, epoch_metric = self._update_metrics(labels, outputs, step_loss)

        step_logs = {"eval_step_loss": step_loss.item(), "eval_epoch_loss": loss_metric,
                     "eval_epoch_metric": epoch_metric}
        return step_logs

    def _update_metrics(self, labels, outputs, step_loss):
        self.loss_metric.update(step_loss.to(self.configs.device))
        loss_metric = self.loss_metric.compute().to("cpu").item()
        epoch_metric = None
        if self.metric is not None:
            self.metric.update(outputs.to(self.configs.device), labels.to(self.configs.device))
            epoch_metric = self.metric.compute().to("cpu").item()
        return loss_metric, epoch_metric

    @torch.no_grad()
    def evaluate(self, model, logs):
        model.eval()
        self.callbacks.on_eval_begin(logs)
        self.metric.reset()
        eval_dataloader = self.get_eval_dataloader()
        if eval_dataloader is None:
            trainer_log.warning("eval_dataloader is None!")
            return logs
        for _, inputs in enumerate(eval_dataloader):
            self.callbacks.on_eval_batch_begin(inputs, logs)
            inputs = [input_.to(self.configs.device) for input_ in inputs]
            step_logs = self.evaluate_step(model, inputs)
            logs.update(step_logs)
            self.callbacks.on_eval_batch_end(tuple(inputs), logs)
            logs["eval_global_step"] += 1
        self.callbacks.on_eval_end(logs)
        return logs

    @torch.no_grad()
    def predict(self, input_):
        self.model.eval()
        return self.model(input_)

    def train_step(self, model, inputs):
        labels = inputs[1]
        outputs = model(inputs[0])
        step_loss = self.compute_loss(labels, outputs)
        step_loss = reduce_value(step_loss, average=True)
        step_loss.backward()
        step_loss = step_loss.detach()

        loss_metric, epoch_metric = self._update_metrics(labels, outputs, step_loss)
        # batch_loss_sum += loss.item()
        # epoch_loss = batch_loss_sum / (i + 1)  # update mean losses
        # if epoch_metric is None:
        #     show_metric = None
        # else:
        #     show_metric = round(epoch_metric.item(), 3)
        # if is_main_process:
        #     train_dataloader.desc = "[epoch {}/{}] loss={}, metric={}".format(epoch + 1,
        #                                                                       int(self.configs.epoch_num),
        #                                                                       round(epoch_loss, 3),
        #                                                                       show_metric)
        # Optimizer step
        optimizer_was_run = True
        self.optimizer.step()
        if optimizer_was_run:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        step_logs = {"step_loss": step_loss.item(), "epoch_loss": loss_metric, "epoch_metric": epoch_metric}
        return step_logs

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

        # self.mean_loss_metric.update(loss)
        # self.mean_loss_metric.compute()
        return loss  # , step_metric

    def push_model_to_hub(self):
        pass

    def add_callback(self, callback):
        self.callbacks.add_callback(callback)

    def set_optimizer(self, optimizer: optim.Optimizer, optimizer_name: str = None):
        """
        set custom optimizer, `optimizer_name` is the optimizer str in training config
        """
        self.override_optimizer = True
        self.configs.optimizer = CUSTOM if optimizer_name is None else optimizer_name
        self.optimizer = optimizer

    def set_loss(self, loss, loss_name=None):
        """
        set custom loss, `loss_name` is the loss str in training config
        """
        self.override_loss = True
        self.configs.loss = CUSTOM if loss_name is None else loss_name
        self.loss = loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, TorchDataSet):
            self.train_dataset = self.train_dataset.dataset
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

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the eval :class:`~torch.utils.data.DataLoader`.
        """
        if self.eval_dataset is None:
            trainer_log.warning("Trainer: eval requires a train_dataset.")
            return None
        if isinstance(self.eval_dataset, TorchDataSet):
            self.eval_dataset = self.eval_dataset.dataset
        if isinstance(self.eval_dataset, IterableDataset):
            return DataLoader(
                self.eval_dataset,
                batch_size=self.configs.eval_batch_size,
            )
        return DataLoader(
            self.eval_dataset,
            batch_size=self.configs.eval_batch_size,
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
        summary_writer_constructor = get_summary_writer_constructor()
        if summary_writer_constructor is not None and self.configs.use_tensorboard:
            self.callbacks.add_callback(
                TensorBoardCallBack(summary_writer_constructor,
                                    log_dir=self.configs.tensorboard_log_dir,
                                    comment=self.configs.tensorboard_comment))
        if self.configs.print_steps is None:
            self.callbacks.add_callback(ProgressBarCallBack(total_epoch_num=self.configs.epoch_num,
                                                            train_dataloader=self.get_train_dataloader()))
        else:
            self.callbacks.add_callback(PrintCallBack(total_epoch_num=self.configs.epoch_num,
                                                      step_frequency=self.configs.print_steps))

    def _create_metric(self):
        self.metric = torchmetrics.Accuracy().to(self.configs.device)
        # self.metric = getattr(torchmetrics, self.configs.metric).to(
        #     self.configs.device)  # todo #get_metric_by_name(self.configs.metric).to(self.configs.device)
        self.loss_metric = torchmetrics.MeanMetric().to(self.configs.device)  # get_metric_by_name("MeanMetric") #todo

    def _create_loss(self):
        if self.override_loss is True:
            return
        self.loss = _construct_loss_from_config(torch.nn.modules.loss, self.configs.loss)

    def _create_optimizer(self):
        if self.override_optimizer is True:
            return
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
        # modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        print(f"Loading from previous checkpoint: {checkpoint_path}")
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
        metric = checkpoint["metric"]
        print("metric = ", metric)
        # if Path(modelcard_path).exists():
        #     self.model_card = ModelCard.load_from_file(modelcard_path)
        #     print(f"Model card is loaded from {modelcard_path}")
            # print("model_card = ", self.model_card.to_dict())
        # else:
        #     trainer_log.warning("model card file not exist.")
        return epoch, loss, metric

    def save(self, path, overwrite=True):
        if not overwrite:
            if Path(path).exists():
                raise FileExistsError("File already exists: ", str(Path(path).resolve()))
        Path(path).mkdir(exist_ok=True)
        checkpoint_path = Path(path).joinpath(CHECKPOINT_NAME)
        modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        trainer_log.info("save checkpoint_path: %s", checkpoint_path)
        optimizer_state_dict = None
        if isinstance(self.optimizer, Optimizer):  # if created
            optimizer_state_dict = self.optimizer.state_dict()
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "loss": (self.loss, self.loss_value),
            "metric": (self.metric, self.metric_value)
        }, checkpoint_path)
        if isinstance(self.model_card, ModelCard):
            self.model_card.save_model_card(modelcard_path)
        else:
            trainer_log.warning("model card is None.")
