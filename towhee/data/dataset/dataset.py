from torch.utils.data.dataset import Dataset


class TowheeDataSet:
    """
    TowheeDataSet is a kind of dataset wrapper, where the `self.dataset` is the true dataset.
    """
    def __init__(self):
        self.framework = None
        self.dataset = None

class TorchDataSet(TowheeDataSet):
    """
    pytorch dataset
    """
    def __init__(self, torchDataSet: Dataset = None):
        super().__init__()
        self.framework = 'torch'
        self.dataset = torchDataSet

    def __len__(self):
        return len(self.dataset)

    def get_framework(self):
        return str(self.framework)
