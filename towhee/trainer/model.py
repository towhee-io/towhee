from torch import nn
from typing import Dict, List, Optional, Tuple

from trainer.modelcard import ModelCard


class Model():
    def __init__(
            self,
            model: nn.Module = None,
            model_card: ModelCard = None
    ):
        self.model = model
        self.model_card = model_card

    def push_to_hub(self):
        pass