from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class TrainingSummary:
    """
    train result summary
    """
    model_name: str
    finetuned_from: Optional[str] = None
    tasks: Optional[Union[str, List[str]]] = None
    dataset: Optional[Union[str, List[str]]] = None
    dataset_tags: Optional[Union[str, List[str]]] = None
    dataset_args: Optional[Union[str, List[str]]] = None
    eval_results: Optional[Dict[str, float]] = None
    eval_lines: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        pass

    def to_model_card(self):
        pass
