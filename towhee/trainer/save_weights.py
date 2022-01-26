from pathlib import Path
import torch


class SaveWeights:
    """
    Args:
        overwrite: if or not overwrite the weights file
    """

    def __init__(self, operator: str = 'NNOperator', overwrite: bool = True):
        self.operator = operator
        self.overwrite = overwrite

    def save_weights(self, model: object, weights_path):
        try:
            if not self.overwrite:
                if Path(weights_path).exists():
                    raise FileExistsError('File already exists: ', str(Path(weights_path).resolve()))
            torch.save(model.state_dict(), weights_path)

        except Exception as e:
            raise e
