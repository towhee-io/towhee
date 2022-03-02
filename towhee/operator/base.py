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

from abc import abstractmethod
from abc import ABC
from enum import Enum

# NotShareable:
#    Stateful & reusable operator.

# NotRusable:
#    Stateful & not reusable operator.

# Shareable:
#    Stateless operator

SharedType = Enum('SharedType', ('NotShareable', 'NotReusable', 'Shareable'))


class Operator(ABC):
    """
    Operator base class, implements __init__ and __call__,

    Examples:
        class AddOperator(Operator):
            def __init__(self, factor: int):
                self._factor = factor

            def __call__(self, num) -> NamedTuple("Outputs", [("sum", int)]):
                Outputs = NamedTuple("Outputs", [("sum", int)])
                return Outputs(self._factor + num)
    """

    @abstractmethod
    def __init__(self):
        """
        Init operator, before a graph starts, the framework will call Operator __init__ function.

        Args:

        Raises:
            An exception during __init__ can terminate the graph run.
        """
        self._key = ''

    @abstractmethod
    def __call__(self):
        """
        The framework calls __call__ function repeatedly for every input data.

        Args:

        Returns:

        Raises:
            An exception during __init__ can terminate the graph run.
        """
        raise NotImplementedError

    @property
    def key(self):
        return self._key

    @property
    def shared_type(self):
        return SharedType.NotShareable

    @key.setter
    def key(self, value):
        self._key = value


class NNOperator(Operator):
    """
    Neural Network related operators that involve machine learning frameworks.

    Args:
        framework (`str`):
            The framework to apply.
    """

    def __init__(self, framework: str = 'pytorch'):
        super().__init__()
        self._framework = framework
        self.operator_name = type(self).__name__
        self.model = None
        self.model_card = None
        self._trainer = None  # Trainer(self.get_model())

    @property
    def framework(self):
        return self._framework

    @framework.setter
    def framework(self, framework: str):
        self._framework = framework

    def get_model(self):
        """
        Get the framework naive model, if an operator need to be trained,
        this method should be overwritten.
        """
        raise NotImplementedError()

    def train(self,
              training_config=None,
              train_dataset=None,
              eval_dataset=None,
              resume_checkpoint_path=None,
              **kwargs):
        """
        Start to train an operator.

        Args:
            training_config (`TrainingConfig`):
                The config of this trainer.
            train_dataset (`Union[Dataset, TowheeDataSet]`):
                Training dataset.
            eval_dataset (`Union[Dataset, TowheeDataSet]`):
                Evaluate dataset.
            resume_checkpoint_path (`str`):
                If resuming training, pass into the path.
            **kwargs (`Any`):
                Keyword Args.
        """
        self.setup_trainer(training_config, train_dataset, eval_dataset, **kwargs)
        self.trainer.train(resume_checkpoint_path)

    @property
    def trainer(self) -> 'Trainer':
        if self._trainer is None:
            self.setup_trainer()
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    def setup_trainer(self, training_config=None,
                      train_dataset=None,
                      eval_dataset=None,
                      train_dataloader=None,
                      eval_dataloader=None,
                      model_card=None
                      ):
        """
        Set up the trainer instance in operator before training and set trainer parameters.
        Args:
            training_config (`TrainingConfig`):
                The config of this trainer.
            train_dataset (`Union[Dataset, TowheeDataSet]`):
                Training dataset.
            eval_dataset (`Union[Dataset, TowheeDataSet]`):
                Evaluate dataset.
            train_dataloader (`Union[DataLoader, Iterable]`):
                If specified, `Trainer` will use it to load training data.
                Otherwise, `Trainer` will build dataloader from train_dataset.
            eval_dataloader (`Union[DataLoader, Iterable]`):
                If specified, `Trainer` will use it to load evaluate data.
                Otherwise, `Trainer` will build dataloader from train_dataset.
            model_card (`ModelCard`):
                Model card contains the training informations.
        Returns:

        """
        from towhee.trainer.trainer import Trainer  # pylint: disable=import-outside-toplevel
        if self._trainer is None:
            self._trainer = Trainer(self.get_model(),
                                    training_config,
                                    train_dataset,
                                    eval_dataset,
                                    train_dataloader,
                                    eval_dataloader,
                                    model_card)
        else:
            if training_config is not None:
                self._trainer.configs = training_config
            if train_dataset is not None:
                self._trainer.train_dataset = train_dataset
            if eval_dataset is not None:
                self._trainer.eval_dataset = eval_dataset
            if train_dataloader is not None:
                self._trainer.train_dataloader = train_dataloader
            if eval_dataloader is not None:
                self._trainer.eval_dataloader = eval_dataloader
            if model_card is not None:
                self._trainer.model_card = model_card

    def load(self, path: str = None):
        """
        Load the model checkpoint into an operator.

        Args:
            path (`str`):
                The folder path containing the model's checkpoints.
        """
        self.trainer.load(path)

    def save(self, path: str, overwrite: bool = True):
        """
        Save the model checkpoint into the path.

        Args:
            path (`str`):
                The folder path containing the model's checkpoints.
            overwrite (`bool`):
                If True, it will overwrite the same name path when existing.

        Raises:
            (`FileExistsError`)
                If `overwrite` is False, when there already exists a path, it will raise Error.
        """
        self.trainer.save(path, overwrite)

    # def change_before_train(self, **kwargs):
    #     pass


class PyOperator(Operator):
    """
    Python function operator, no machine learning frameworks involved.
    """

    def __init__(self):
        super().__init__()
        pass
