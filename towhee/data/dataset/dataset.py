from torch.utils.data.dataset import Dataset
from torchvision import datasets


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


def dataset(name, *args, **kwargs) -> TorchDataSet:
    """
    mapping to a dataset by name, and pass into the custom params
    """
    dataset_construct_map = {
        'mnist': datasets.MNIST,
        'cifar10': datasets.cifar.CIFAR10,
        'fake': datasets.FakeData
        # 'imdb': IMDB  # ,()
    }
    torch_dataset = dataset_construct_map[name](*args, **kwargs)
    return TorchDataSet(torch_dataset)
