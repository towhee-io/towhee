import torch


class LoadWeights:
    """
    Args:
        device: 'cpu'
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def load_weights(self, model, weight_path):
        device = self.device
        weights = torch.load(weight_path, map_location=device)
        assert isinstance(weights, dict)
        try:
            model.load_state_dict(weights, strict=False)
            return model
        except Exception as e:
            raise e
