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

"""
Trainer wrapper for cnn and transformer task.
"""

import vit_trainer
import cnn_trianer


class TorchTrainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch.

    Args:
        is_transformer (:obj:`bool`):
            If the trainer is a transformer trainer. Will default to :obj:`False` if it is a cnn model based trainer
            and not a transformer trainer, :obj:`True` otherwise.
        model (:class:`~towhee.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

            .. note::

                :class:`~towhee.Trainer` is optimized to work with the :class:`~towhee.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the  Transformers models.
        args (:class:`~towhee.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~towhee.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset` or :obj:`torch.utils.data.dataset.IterableDataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.

            Note that if it's a :obj:`torch.utils.data.dataset.IterableDataset` with some randomization and you are
            training in a distributed fashion, your iterable dataset should either use a internal attribute
            :obj:`generator` that is a :obj:`torch.Generator` for the randomization that must be identical on all
            processes (and the Trainer will manually set the seed of this :obj:`generator` at each epoch) or have a
            :obj:`set_epoch()` method that internally sets the seed of the RNGs used.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~towhee.Trainer.train` will start from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be
            able to choose different architectures according to hyper parameters (such as layer count, sizes of inner
            layers, dropout probabilities etc).
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~towhee.EvalPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~towhee.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a
          :class:`~transformers.PreTrainedModel` subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. 
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device.
        - **is_in_train** -- Whether or not a model is currently running ``train`` (e.g. when ``evaluate`` is called
          while in ``train``)

    """

    def __init__(
            self,
            is_transformer: bool = False,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            tokenizer: Optional[PreTrainedTokenizerBase] = None
    ):
        if is_transformer:
            self.cnn = vit_trainer(model, args, train_dataset, eval_dataset, tokenizer,
                                   model_init, compute_metrics, callbacks, optimizers)
        else:
            self.transformer = cnn_trianer(model, args, train_dataset, eval_dataset, model_init,
                                           compute_metrics, callbacks, optimizers)
