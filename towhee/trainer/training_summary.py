from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from towhee.trainer.utils import logging
from towhee.trainer.training_config import TrainingConfig

logger = logging.get_logger(__name__)



@dataclass
class TrainingSummary:
    """
    train result summary
    """
    model_name: str
    # tags: Optional[Union[str, List[str]]] = None
    finetuned_from: Optional[str] = None
    tasks: Optional[Union[str, List[str]]] = None
    dataset: Optional[Union[str, List[str]]] = None
    # dataset_tags: Optional[Union[str, List[str]]] = None
    # dataset_args: Optional[Union[str, List[str]]] = None
    eval_results: Optional[Dict[str, float]] = None
    # eval_lines: Optional[List[str]] = None
    # hyperparameters: Optional[Dict[str, Any]] = None
    train_config: Optional[TrainingConfig] = None

    def __post_init__(self):
        pass

    def from_train_config(self, train_config):
        pass


# if __name__ == '__main__':
#     ts = TrainingSummary(model_name='model3', tags=['tag1', 'tag2'],tasks='task1', dataset='dataset111',
#                          eval_results={'e1':11,'e2':22}, dataset_tags='dataset_tag1')
#     print(ts)
