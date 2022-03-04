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

import math
import os
import torch
import torch.distributed as dist

from typing import Union, Dict, Any, Optional
from pathlib import Path
from collections.abc import Iterable
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.multiprocessing import Process
from towhee.data.dataset.dataset import TowheeDataSet, TorchDataSet

from towhee.trainer.callback import TensorBoardCallBack, ProgressBarCallBack, PrintCallBack, ModelCheckpointCallback, \
    EarlyStoppingCallback, TrainerControl, Callback
from towhee.trainer.metrics import get_metric_by_name
from towhee.trainer.modelcard import ModelCard, MODEL_CARD_NAME
from towhee.trainer.utils.trainer_utils import STATE_CHECKPOINT_NAME, MODEL_NAME, set_seed, reduce_value, \
    is_main_process, send_to_device
from towhee.trainer.training_config import TrainingConfig
from towhee.utils.log import trainer_log
from towhee.trainer.optimization.optimization import get_scheduler
from towhee.trainer.callback import CallbackList, _get_summary_writer_constructor

WEIGHTS_NAME = "pytorch_model.bin"
TEMP_INIT_WEIGHTS = "./temp_init_weights.pt"
NAME = "name_"
CUSTOM = "custom_"
no_option_list = ["no", "null", "None", None, False]


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


def _construct_scheduler_from_config(module: Any, config: Union[str, Dict]):
    """
    construct from the config, the config can be class name as a `str`, or a dict containing the construct parameters.
    """
    instance = None
    if isinstance(config, str):
        construct_name = getattr(module, config)
        instance = construct_name()
    elif isinstance(config, Dict):
        scheduler_construct_name = config[NAME]
        construct_name = getattr(module, scheduler_construct_name)
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
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            instance = construct_name(trainable_params, **kwargs)
    return instance


def freeze_bn(model):
    classname = model.__class__.__name__
    if classname.find("BatchNorm") != -1:
        model.eval()


class Trainer:
    """
    A `Trainer` is used to train a pytorch model.

    Args:
        model (`nn.Module`):
            A pytorch model.
        training_config (`TrainingConfig`):
            A `TrainingConfig` instance can be loaded from a yaml file.
        train_dataset (`Union[Dataset, TowheeDataSet]`):
            It can be a kind of `torch.utils.data.dataset.Dataset` or `TowheeDataSet`
        eval_dataset (`Union[Dataset, TowheeDataSet]`):
            The same as `train_dataset`, and it is not strictly necessary if you do not want to eval.
        train_dataloader (`Union[DataLoader, Iterable]`):
            When the `train_dataloader` is passed in, trainer will use it to load data
            instead of constructing by input dataset.
        eval_dataloader (`Union[DataLoader, Iterable]`):
            The same as `train_dataloader`, and it is not strictly necessary also.
        model_card (`ModelCard`):
            Model card may contain other information of a model, and it is not strictly necessary.

    Examples:
        >>> import torch
        >>> import torchvision.models as models
        >>> from towhee.trainer.trainer import Trainer
        >>> from towhee.trainer.training_config import TrainingConfig
        >>> from towhee import dataset
        >>> from torchvision import transforms
        >>> import warnings
        >>> warnings.filterwarnings("ignore")
        >>> model = models.resnet18()
        >>> fake_transform = transforms.Compose([transforms.ToTensor()])
        >>> train_data = dataset('fake', size=2, transform=fake_transform)
        >>> val_data = dataset('fake', size=1, transform=fake_transform)
        >>> training_config = TrainingConfig(output_dir="train_res",
        ...                                  tensorboard=None,
        ...                                  epoch_num=2,
        ...                                  batch_size=1,
        ...                                  dataloader_num_workers=0,
        ...                                  print_steps=1)
        >>> trainer = Trainer(model, training_config, train_dataset=train_data, eval_dataset=val_data)
        >>> type(trainer)
        <class 'towhee.trainer.trainer.Trainer'>
        >>> trainer.train()  # Some values below are not necessarily reproducible
        2022-03-01 19:01:54,314 - 8601085440 - trainer.py-trainer:324 - WARNING: TrainingConfig(...)
        epoch=1/2, global_step=1, epoch_loss=7.107283592224121, epoch_metric=0.0
        epoch=1/2, global_step=2, epoch_loss=6.959554195404053, epoch_metric=0.0
        epoch=1/2, eval_global_step=0, eval_epoch_loss=6.694866180419922, eval_epoch_metric=0.0
        epoch=2/2, global_step=3, epoch_loss=6.165490627288818, epoch_metric=0.0
        epoch=2/2, global_step=4, epoch_loss=6.197325706481934, epoch_metric=0.0
        epoch=2/2, eval_global_step=1, eval_epoch_loss=6.0876030921936035, eval_epoch_metric=0.0
        >>> trainer.train(resume_checkpoint_path="train_res/epoch_1")
        2022-03-01 19:01:57,004 - 8601085440 - trainer.py-trainer:324 - WARNING: TrainingConfig(...)
        epoch=2/2, global_step=1, epoch_loss=6.165490627288818, epoch_metric=0.0
        epoch=2/2, global_step=2, epoch_loss=6.277336120605469, epoch_metric=0.0
        epoch=2/2, eval_global_step=0, eval_epoch_loss=6.097333908081055, eval_epoch_metric=0.0
        >>> trainer.save(path="another_save_path")
        >>> trainer.load(path="another_save_path")
        >>> trainer.epoch
        2
        >>> model = models.resnet18()
        >>> inputs = [torch.randn(1, 3, 224, 224), torch.Tensor([1]).type(torch.LongTensor)]
        >>> model.eval()  # turn on eval mode
        ResNet(
        ...
        )
        >>> trainer.evaluate_step(model, inputs)
        {'eval_step_loss': 7.10837459564209, 'eval_epoch_loss': 6.350093841552734, 'eval_epoch_metric': 0.0}
        >>> trainer.update_metrics(model, inputs, torch.Tensor(1))
        (5.2800750732421875, 0.0)
        >>> trainer.compute_metric(model, inputs)
        0.0
        >>> trainer.evaluate(model, {"epoch": 1, "eval_global_step": 1})
        epoch=1/2, eval_global_step=1, eval_epoch_loss=5.654547214508057, eval_epoch_metric=0.0
        {'epoch': 1, 'eval_global_step': 2, 'eval_step_loss': 7.526906967163086, 'eval_epoch_loss': 5.654547214508057, 'eval_epoch_metric': 0.0}
        >>> trainer.predict(inputs[0]).shape
        torch.Size([1, 1000])
        >>> model.train()  # turn on train mode
        ResNet(
        ...
        )
        >>> trainer.train_step(model, inputs)
        {'step_loss': 7.10837459564209, 'epoch_loss': 5.862236976623535, 'epoch_metric': 0.0}
        >>> trainer.compute_loss(model, inputs)
        tensor(7.1084, grad_fn=<NllLossBackward>)
        >>>
        >>> from towhee.trainer.callback import Callback
        >>> from typing import Dict
        >>> class MyCallback(Callback):
        ...     def on_eval_begin(self, logs: Dict) -> Dict:
        ...         print("on_eval_begin...")
        ...
        >>> my_callback = MyCallback()
        >>> trainer.add_callback(my_callback)
        >>> trainer.evaluate(model, logs={"epoch": 1, "eval_global_step": 1})
        on_eval_begin...
        epoch=1/2, eval_global_step=1, eval_epoch_loss=6.070321083068848, eval_epoch_metric=0.0
        {'epoch': 1, 'eval_global_step': 2, 'eval_step_loss': 7.526906967163086, 'eval_epoch_loss': 6.070321083068848, 'eval_epoch_metric': 0.0}
    """

    def __init__(
            self,
            model: nn.Module = None,
            training_config: TrainingConfig = None,
            train_dataset: Union[Dataset, TowheeDataSet] = None,
            eval_dataset: Union[Dataset, TowheeDataSet] = None,
            train_dataloader: Union[DataLoader, Iterable] = None,
            eval_dataloader: Union[DataLoader, Iterable] = None,
            model_card: ModelCard = None
    ):

        if training_config is None:
            output_dir = "tmp_trainer"
            trainer_log.warning("No `TrainingConfig` passed.")
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
        self.lr_value = self.configs.lr
        self.metric = None
        self.loss = None
        self.override_loss = False
        self.loss_value = 0.0
        self.callbacks = CallbackList()
        self.loss_metric = None
        self.metric_value = 0.0
        self.epoch = 0
        self.train_dataloader = train_dataloader
        self.train_sampler = None
        self.eval_dataloader = eval_dataloader
        self.distributed = False

        os.makedirs(self.configs.output_dir, exist_ok=True)
        if not isinstance(self.model_card, ModelCard):
            self.model_card = ModelCard()

        if self.model_card.model_name is None:
            self.model_card.model_name = type(self.model).__name__
        self.model_card.model_architecture = str(self.model)
        self.model_card.training_config = self.configs

    def train(self, resume_checkpoint_path: Optional[str] = None):
        """
        Start to train.
        Args:
            resume_checkpoint_path (`int`):
                The path to start resume training.
        """
        if self.configs.device_str == "cuda":
            self.distributed = True
            self._spawn_train_process(resume_checkpoint_path)
        else:
            self.distributed = False
            self.run_train(resume_checkpoint_path)

    def _spawn_train_process(self, resume_checkpoint_path: Optional[str]):
        # world_size = torch.cuda.device_count()
        # mp.spawn(self.run_train,
        #          args=(world_size, resume_checkpoint_path),
        #          nprocs=world_size,  # opt.world_size,
        #          join=True)
        process_list = []
        world_size = self.configs.n_gpu
        if world_size < 1:
            trainer_log.warning("when `device_str` is `cuda`, `n_gpu` must be a positive int number.")
        for rank in range(world_size):
            process = Process(target=self.run_train, args=(resume_checkpoint_path, rank, world_size))
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()

    def _init_distributed(self, rank: int, world_size: int):
        if self.distributed:
            if torch.cuda.is_available() is False:
                raise EnvironmentError("not find GPU device for training.")
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            trainer_log.warning("_init_distributed(), rank=%s", rank)
            torch.cuda.set_device(rank)
            dist_backend = "nccl"
            dist_url = "env://"
            trainer_log.warning("| distributed init (rank %s): %s", rank, dist_url)
            dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                    world_size=world_size, rank=rank)
            dist.barrier()

    def _load_before_train(self, resume_checkpoint_path: Optional[str], rank: Optional[int]):
        # sync_bn = self.configs.sync_bn
        if resume_checkpoint_path is not None:
            # weights_dict = torch.load(weights_path, map_location=device)
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if model.state_dict()[k].numel() == v.numel()}
            # model.load_state_dict(load_weights_dict, strict=False)
            self.load(resume_checkpoint_path)
        else:  # if using multi gpu and not resume, must keep model replicas in all processes are the same
            if self.distributed:
                # checkpoint_path = os.path.join(tempfile.gettempdir(), TEMP_INIT_WEIGHTS)
                checkpoint_path = TEMP_INIT_WEIGHTS
                if rank == 0:
                    torch.save(self.model.state_dict(), checkpoint_path)
                dist.barrier()
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.configs.device))
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
            # if sync_bn:
            #     self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.configs.device)

    def _create_logs(self):
        logs = {"global_step": 0, "epoch": self.epoch}
        if self.configs.eval_strategy not in no_option_list:
            logs["eval_global_step"] = 0
        return logs

    def prepare_inputs(self, inputs: Any):
        return send_to_device(inputs, self.configs.device)

    def run_train(self, resume_checkpoint_path: str = None, rank: int = None, world_size: int = None):
        """
        Main training entry point.
        It is not recommended for users to use it unless over rewriting Trainer.
        Instead, it is recommended to use `trainer.train()` to start training.

        Args:
            resume_checkpoint_path (`str`):
                Last checkpoint path.
            rank (`int`):
                Process rank when using multi gpus.
            world_size (`int`):
                Total processes count.
        """
        set_seed(self.configs.seed)
        self._init_distributed(rank, world_size)

        self.model = self.model.to(self.configs.device)
        model = self.model
        self.trainercontrol = TrainerControl()
        self._load_before_train(resume_checkpoint_path, rank)
        # Keeping track whether we can can len() on the dataset or not
        # train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = self.configs.train_batch_size
        # if train_dataset_is_sized:
        num_update_steps_per_epoch = len(train_dataloader)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        train_last_epoch = self.epoch
        num_train_epochs = math.ceil(self.configs.epoch_num - train_last_epoch)
        num_train_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)

        self.setup_before_train(num_training_steps=num_train_steps, init_lr=self.lr_value)

        trainer_log.info("***** Running training *****")
        trainer_log.info("  Num Epochs = %d", num_train_epochs)
        trainer_log.info("  Total train batch size  = %d", total_train_batch_size)
        trainer_log.info("****************************")

        if is_main_process():
            trainer_log.warning(self.configs)

        logs = self._create_logs()
        self.callbacks.on_train_begin(logs)

        for epoch in range(train_last_epoch + 1, self.configs.epoch_num + 1):
            self.epoch = logs["epoch"] = epoch
            self.set_train_mode(model)
            self._reset_controller()
            self.optimizer.zero_grad()
            if self.distributed:
                self.train_sampler.set_epoch(self.epoch)
            # batch_loss_sum = 0.0
            self.callbacks.on_epoch_begin(self.epoch, logs)
            self.loss_metric.reset()
            self.metric.reset()
            for step, inputs in enumerate(train_dataloader):
                self.callbacks.on_train_batch_begin(inputs, logs)
                inputs = self.prepare_inputs(inputs)
                step_logs = self.train_step(model, inputs)  # , train_dataloader)
                logs["lr"] = self.lr_scheduler.get_lr()[0]
                logs["global_step"] += 1
                logs.update(step_logs)
                self.callbacks.on_train_batch_end(tuple(inputs), logs)
                self._may_evaluate(model, logs, step)
            self._may_evaluate(model, logs)
            self.callbacks.on_epoch_end(self.epoch, logs)
            self.loss_value = logs["epoch_loss"]
            self.metric_value = logs["epoch_metric"]
            self.lr_value = logs["lr"]
            self._create_training_summary(
                finetuned_from=resume_checkpoint_path,
                resumed_from_epoch=train_last_epoch if train_last_epoch != 0 else None,
                num_train_epochs=self.epoch - train_last_epoch,
                current_epoch=self.epoch,
                end_lr=self.lr_value,
                loss={"type": self.configs.loss, "value": round(self.loss_value, 3)},
                metric={"type": self.configs.metric, "value": round(self.metric_value, 3)}
            )
            if self.trainercontrol.should_training_stop:
                break
            if self.trainercontrol.should_save:
                self.save(
                    path=os.path.join(self.configs.output_dir, "epoch_" + str(self.epoch)),
                    overwrite=self.configs.overwrite_output_dir
                )
        trainer_log.info("\nTraining completed.\n")

        self._cleanup_distributed(rank)
        self.callbacks.on_train_end(logs)

        self.save(
            path=os.path.join(self.configs.output_dir, "final_epoch"),
            overwrite=self.configs.overwrite_output_dir
        )

    def set_train_mode(self, model: nn.Module):
        """
        Convert the model to training mode.

        Args:
            model (`nn.Module`):
        """
        model.train()
        if self.configs.freeze_bn:
            model.apply(freeze_bn)

    def _create_training_summary(self, **kwargs):
        training_summary = dict(kwargs)
        self.model_card.training_summary = training_summary

    def _may_evaluate(self, model: nn.Module, logs: dict, step: int = -1):
        if step != -1:  # step end
            if self.configs.eval_strategy in ["step", "steps"]:
                assert self.configs.eval_steps > 0, "self.configs.eval_steps must be a positive int number"
            if self.configs.eval_strategy in ["step", "steps"] and step % self.configs.eval_steps == 0:
                eval_logs = self.evaluate(model, logs)
                logs.update(eval_logs)
        else:  # epoch end
            if self.configs.eval_strategy in ["epoch", "eval_epoch"]:
                eval_logs = self.evaluate(model, logs)
                logs.update(eval_logs)

    @torch.no_grad()
    def evaluate_step(self, model: nn.Module, inputs: Any) -> dict:
        """
        One batch step when evaluating.

        Args:
            model (`nn.Module`):
                Pytorch model.
            inputs (`Any`):
                Input Tensor or any kind of collection made up by tensors.

        Returns:
            (`dict`)
                Evaluate logs dict.
        """
        inputs = self.prepare_inputs(inputs)
        step_loss = self.compute_loss(model, inputs)
        step_loss = reduce_value(step_loss, average=True)
        step_loss = step_loss.detach()

        loss_metric, epoch_metric = self.update_metrics(model, inputs, step_loss, training=False)

        step_logs = {"eval_step_loss": step_loss.item(), "eval_epoch_loss": loss_metric,
                     "eval_epoch_metric": epoch_metric}
        return step_logs

    @torch.no_grad()
    def update_metrics(self, model: nn.Module, inputs: Any, step_loss: torch.Tensor, training: bool = True) -> tuple:
        """
        Update the loss and metric in one epoch.
        When restart a new epoch, the epoch_loss and epoch metric will be clear.

        Args:
            model (`nn.Module`):
                Pytorch model.
            inputs (`Any`):
                Torch tensor or any kind of collection made up by tensors.
            step_loss (`torch.Tensor`):
                One batch step loss.
            training (`bool`):
                Whether it's training mode now.

        Returns:
            (`tuple`)
                Epoch loss and epoch metric.
        """
        self.loss_metric.update(send_to_device(step_loss, self.configs.device))
        loss_metric = self.loss_metric.compute().item()
        if self.configs.eval_strategy == "eval_epoch" and training:
            epoch_metric = 0
        else:
            epoch_metric = self.compute_metric(model, inputs)
        return loss_metric, epoch_metric

    @torch.no_grad()
    def compute_metric(self, model: nn.Module, inputs: Any) -> float:
        """
        Compute the step metric.
        It is recommended to subclass `Trainer` and override this method when deal with custom metric in custom task.
        When it is overridden, another method `compute_loss()` often needs to be overridden.

        Args:
            model (`nn.Module`):
                Pytorch model.
            inputs (`Any`):
                Input tensor collection.

        Returns:
            (`float`)
                Epoch average metric.
        """
        model.eval()
        epoch_metric = None
        labels = inputs[1]
        outputs = model(inputs[0])
        if self.metric is not None:
            self.metric.update(send_to_device(outputs, self.configs.device),
                               send_to_device(labels, self.configs.device))
            epoch_metric = self.metric.compute().item()
        return epoch_metric

    @torch.no_grad()
    def evaluate(self, model: nn.Module, logs: dict) -> dict:
        """
        Evaluate the model.

        Args:
            model (`nn.Module`):
                Pytorch model.
            logs (`dict`):
                Logs dict.

        Returns:
            (`dict`)
                The new logs dict with evaluate values.
        """
        model.eval()
        self.callbacks.on_eval_begin(logs)
        self.metric.reset()
        eval_dataloader = self.get_eval_dataloader()
        if eval_dataloader is None:
            trainer_log.warning("eval_dataloader is None!")
            return logs
        for _, inputs in enumerate(eval_dataloader):
            self.callbacks.on_eval_batch_begin(inputs, logs)
            inputs = send_to_device(inputs, self.configs.device)
            step_logs = self.evaluate_step(model, inputs)
            logs.update(step_logs)
            self.callbacks.on_eval_batch_end(tuple(inputs), logs)
            logs["eval_global_step"] += 1
        self.callbacks.on_eval_end(logs)
        return logs

    @torch.no_grad()
    def predict(self, inputs: Any) -> Any:
        """
        Do prediction. The eval mode model passes in the input value and get the outputs.

        Args:
            inputs (`Any`):
                Inference inputs by the model.

        Returns:
            (`Any`)
                Output result.
        """
        self.model.eval()
        return self.model(inputs)

    def train_step(self, model: nn.Module, inputs: Any) -> dict:
        """
        The training batch step.
        It contains computing step loss, loss backpropagation, doing step optimization, computing metric.

        Args:
            model (`nn.Module`):
                Pytorch model.
            inputs (`Any`):
                Pytorch tensor or tensor collection.

        Returns:
            (`dict`)
                Step logs which contains the step loss and metric infos.
        """
        step_loss = self.compute_loss(model, inputs)
        step_loss = reduce_value(step_loss, average=True)
        step_loss.backward()
        step_loss = step_loss.detach()

        loss_metric, epoch_metric = self.update_metrics(model, inputs, step_loss, training=True)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        step_logs = {"step_loss": step_loss.item(), "epoch_loss": loss_metric, "epoch_metric": epoch_metric}
        return step_logs

    def _cleanup_distributed(self, rank: int):
        if self.distributed:
            if rank == 0:
                if os.path.exists(TEMP_INIT_WEIGHTS) is True:
                    os.remove(TEMP_INIT_WEIGHTS)
            dist.destroy_process_group()

    def compute_loss(self, model: nn.Module, inputs: Any):
        """
        Compute the step loss.
        It is recommended to subclass `Trainer` and override this method when deal with custom loss in custom task.
        When it is overridden, another method `compute_metric()` often needs to be overridden.

        Args:
            model (`nn.Module`):
                Pytorch model.
            inputs (`Any`):
                Model inputs when training.

        Returns:
            (`Any`)
                Loss values with `grad_fn`.
        """
        self.set_train_mode(model)
        labels = inputs[1]
        outputs = model(inputs[0])
        loss = self.loss(outputs, labels)
        return loss

    def push_model_to_hub(self):
        # todo
        pass

    def add_callback(self, callback: Callback, singleton: bool = True):
        """
        Users can add their custom callbacks into the trainer.

        Args:
            callback (`Callback`):
                Custom callback.
            singleton (`bool`):
                Whether this kind of callback is singleton.
                When singleton, the same class instance in `trainer.callbacks` will be replaced.
        """
        self.callbacks.add_callback(callback, singleton)

    def set_optimizer(self, optimizer: optim.Optimizer, optimizer_name: str = None):
        """
        Set custom optimizer

        Args:
            optimizer (`optim.Optimizer`):
                User's custom optimizer instance.
            optimizer_name (`str`):
                The optimizer string in training config, if it is `None`, the string will be a default value.

        Examples:
            >>> from towhee.trainer.trainer import Trainer
            >>> from typing import Optional, Callable
            >>> from torch import optim
            >>> import torchvision.models as models
            >>> model = models.resnet18()
            >>> trainer = Trainer(model)
            2022-03-01 17:22:52,306 - 8614221312 - trainer.py-trainer:173 - WARNING: No `TrainingConfig` passed.
            >>> class MyOptimizer(optim.Optimizer):
            ...     def step(self, closure: Optional[Callable[[], float]]=...) -> Optional[float]:
            ...         print('my step...')
            ...
            >>> my_optimizer = MyOptimizer(model.parameters(), defaults={})
            >>> trainer.set_optimizer(my_optimizer)
            >>> type(trainer.optimizer)
            <class '__main__.MyOptimizer'>
        """
        self.override_optimizer = True
        self.configs.optimizer = CUSTOM if optimizer_name is None else optimizer_name
        self.optimizer = optimizer

    def set_loss(self, loss: Any, loss_name: str = None):
        """
        Set custom loss

        Args:
            loss (`Any`):
                User's custom loss instance.
            loss_name (`str`):
                The loss string in training config, if it is `None`, the string will be a default value.

        Examples:
            >>> from towhee.trainer.trainer import Trainer
            >>> import torchvision.models as models
            >>> import torch
            >>> model = models.resnet18()
            >>> trainer = Trainer(model)
            2022-03-01 17:34:36,873 - 8605304320 - trainer.py-trainer:173 - WARNING: No `TrainingConfig` passed.
            >>> class MyTripletLossFunc(torch.nn.Module):
            ...     def forward(self):
            ...         print('forward...')
            ...         return 0
            ...
            >>> my_loss = MyTripletLossFunc()
            >>> trainer.set_loss(my_loss)
            >>> type(trainer.loss)
            <class '__main__.MyTripletLossFunc'>
        """
        self.override_loss = True
        self.configs.loss = CUSTOM if loss_name is None else loss_name
        self.loss = loss

    def _get_num_workers(self):
        if self.configs.dataloader_num_workers == -1:
            num_workers = min([os.cpu_count(), self.configs.batch_size if self.configs.batch_size > 1 else 0, 8])
        else:
            num_workers = self.configs.dataloader_num_workers
        if is_main_process():
            trainer_log.info("num_workers=%s", num_workers)
        return num_workers

    def get_train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader.

        Returns:
            ('Optional[DataLoader]')
                The dataloader to fit data to train the model.
        """
        if self.train_dataloader is not None:
            return self.train_dataloader
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, TorchDataSet):
            self.train_dataset = self.train_dataset.dataset
        num_workers = self._get_num_workers()
        # if isinstance(self.train_dataset, IterableDataset):
        #     return DataLoader(
        #         self.train_dataset,
        #         batch_size=self.configs.train_batch_size,
        #     )
        if not self.distributed:
            return DataLoader(
                self.train_dataset,
                batch_size=self.configs.train_batch_size,
                shuffle=True,
                num_workers=num_workers,  # self.configs.dataloader_num_workers,
                pin_memory=self.configs.dataloader_pin_memory,
                drop_last=self.configs.dataloader_drop_last
            )
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            train_batch_sampler = torch.utils.data.BatchSampler(
                self.train_sampler, self.configs.batch_size, drop_last=True)
            return torch.utils.data.DataLoader(self.train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               num_workers=num_workers,  # self.configs.dataloader_num_workers,
                                               pin_memory=self.configs.dataloader_pin_memory,
                                               )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """
        Get the eval dataloader.

        Returns:
            (`Optional[DataLoader]`)
                The dataloader to fit data to eval the model.
        """
        if self.eval_dataloader is not None:
            return self.eval_dataloader
        if self.eval_dataset is None:
            trainer_log.warning("Trainer: eval requires a train_dataset.")
            return None
        if isinstance(self.eval_dataset, TorchDataSet):
            self.eval_dataset = self.eval_dataset.dataset
        # if isinstance(self.eval_dataset, IterableDataset):
        #     return DataLoader(
        #         self.eval_dataset,
        #         batch_size=self.configs.eval_batch_size,
        #     )
        num_workers = self._get_num_workers()
        if not self.distributed:
            return DataLoader(
                self.eval_dataset,
                batch_size=self.configs.eval_batch_size,
                num_workers=num_workers,  # self.configs.dataloader_num_workers,
                pin_memory=self.configs.dataloader_pin_memory,
                drop_last=self.configs.dataloader_drop_last
            )
        else:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_dataset)
            eval_batch_sampler = torch.utils.data.BatchSampler(
                eval_sampler, self.configs.batch_size, drop_last=True)
            return torch.utils.data.DataLoader(self.eval_dataset,
                                               batch_sampler=eval_batch_sampler,
                                               num_workers=num_workers,  # self.configs.dataloader_num_workers,
                                               pin_memory=self.configs.dataloader_pin_memory,
                                               )

    def setup_before_train(self, num_training_steps: int, init_lr: float):
        """
        Setup some configs before training.

        Args:
            num_training_steps (`int`):
                All training steps in all training loops.
            init_lr (`float`):
                Start learning rate.
        """
        self._create_optimizer(init_lr=init_lr)
        self._create_loss()
        self._create_metric()
        self._create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        self._create_callbacks()

    def _create_callbacks(self):
        # print or progressbar
        if self.configs.print_steps is None:
            self.callbacks.add_callback(ProgressBarCallBack(total_epoch_num=self.configs.epoch_num,
                                                            train_dataloader=self.get_train_dataloader()))
        else:
            self.callbacks.add_callback(PrintCallBack(total_epoch_num=self.configs.epoch_num,
                                                      step_frequency=self.configs.print_steps))
        # early stop
        if self.configs.early_stopping not in no_option_list:
            self.callbacks.add_callback(EarlyStoppingCallback(self.trainercontrol, **self.configs.early_stopping))
        # save checkpoint
        if self.configs.model_checkpoint not in no_option_list:
            self.callbacks.add_callback(ModelCheckpointCallback(self.trainercontrol, **self.configs.model_checkpoint))
        # tensorboard
        if self.configs.tensorboard not in no_option_list:
            summary_writer_constructor = _get_summary_writer_constructor()
            if summary_writer_constructor is not None:
                self.callbacks.add_callback(
                    TensorBoardCallBack(summary_writer_constructor,
                                        **self.configs.tensorboard))

    def _create_metric(self):
        self.metric = get_metric_by_name(self.configs.metric)
        self.metric.to(self.configs.device)
        self.loss_metric = get_metric_by_name("MeanMetric")
        self.loss_metric.to(self.configs.device)

    def _create_loss(self):
        if self.override_loss is True:
            return
        self.loss = _construct_loss_from_config(torch.nn.modules.loss, self.configs.loss)

    def _create_optimizer(self, init_lr: float):
        if self.override_optimizer is True:
            return
        self.optimizer = _construct_optimizer_from_config(
            optim,
            self.configs.optimizer,
            model=self.model,
        )
        for param in self.optimizer.param_groups:
            param.setdefault("initial_lr", init_lr)
        self.optimizer.lr = init_lr

    def _create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if isinstance(self.configs.lr_scheduler_type, str):
            self.lr_scheduler = get_scheduler(
                self.configs.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        else:
            self.configs.lr_scheduler_type["optimizer"] = optimizer
            self.lr_scheduler = _construct_scheduler_from_config(torch.optim.lr_scheduler,
                                                                 self.configs.lr_scheduler_type)
        return self.lr_scheduler

    def get_warmup_steps(self, num_training_steps: int) -> int:
        """
        Get number of steps used for a linear warmup.

        Args:
            num_training_steps (`int`):
                All training steps when training.

        Returns:
            (`int`)
                Warmup steps.
        """
        warmup_steps = (
            self.configs.warmup_steps if self.configs.warmup_steps > 0 else math.ceil(
                num_training_steps * self.configs.warmup_ratio)
        )
        return warmup_steps

    def load(self, path: str):
        """
        Load a model from the path.

        Args:
            path (`str`):
                The folder path containing the model's checkpoints.
        """
        state_path = Path(path).joinpath(STATE_CHECKPOINT_NAME)
        model_path = Path(path).joinpath(MODEL_NAME)
        # modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
        trainer_log.info("Loading from previous checkpoint: %s", model_path)
        state_checkpoint = torch.load(state_path, map_location=self.configs.device)
        model_checkpoint = torch.load(model_path, map_location=self.configs.device)
        self.model.load_state_dict(model_checkpoint)
        if isinstance(self.optimizer, Optimizer) and state_checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(state_checkpoint["optimizer_state_dict"])
        if self.lr_scheduler and state_checkpoint["lr_scheduler_state_dict"]:
            self.lr_scheduler.load_state_dict(state_checkpoint["lr_scheduler_state_dict"])
        if "end_lr" not in state_checkpoint:
            return 0
        self.lr_value = state_checkpoint["end_lr"]
        if "epoch" not in state_checkpoint:
            return 0
        self.epoch = state_checkpoint["epoch"]
        self.loss_value = state_checkpoint["loss_value"]
        self.metric_value = state_checkpoint["metric_value"]

    def save(self, path, overwrite=True):
        """
        Save the checkpoint information in a folder.

        Args:
            path (`str`):
                The folder path containing the model's checkpoints.
            overwrite (`bool`):
                If True, it will overwrite the same name path when existing.

        Raises:
            (`FileExistsError`)
                If `overwrite` is False, when there already exists a path, it will raise Error.
        """
        if is_main_process():
            if not overwrite:
                if Path(path).exists():
                    raise FileExistsError("File already exists: ", str(Path(path).resolve()))
            Path(path).mkdir(exist_ok=True)
            state_path = Path(path).joinpath(STATE_CHECKPOINT_NAME)
            model_path = Path(path).joinpath(MODEL_NAME)
            modelcard_path = Path(path).joinpath(MODEL_CARD_NAME)
            trainer_log.info("save model_path: %s", model_path)
            optimizer_state_dict = None
            lr_scheduler_state_dict = None
            if isinstance(self.optimizer, Optimizer):  # if created
                optimizer_state_dict = self.optimizer.state_dict()
            if self.lr_scheduler is not None:
                lr_scheduler_state_dict = self.lr_scheduler.state_dict()
            torch.save({
                "epoch": self.epoch,
                "optimizer_state_dict": optimizer_state_dict,
                "lr_scheduler_state_dict": lr_scheduler_state_dict,
                "loss_value": self.loss_value,
                "metric_value": self.metric_value,
                "end_lr": self.lr_value
            }, state_path)
            torch.save(self.model.state_dict(), model_path)
            if isinstance(self.model_card, ModelCard):
                self.model_card.save_model_card(modelcard_path)
            else:
                trainer_log.warning("model card is None.")

    def _reset_controller(self):
        self.trainercontrol.should_save = False
        self.trainercontrol.should_training_stop = False
        self.trainercontrol.should_log = False
        self.trainercontrol.should_evaluate = False
        self.trainercontrol.should_epoch_stop = False
