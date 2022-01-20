from torch.utils.data.dataset import Dataset
# from torchtext.datasets
from torchvision import datasets
# from torchtext.datasets import IMDB


class TowheeDataSet:
    """
    TowheeDataSet有点类似在常见的框架上包装了一层，类似一个wrapper，里面有个成员变量dataset才是真正的dataset，
    去继承它的可以是torch、tf等各类框架的dataset，
    包装的目的是为了集成常见的框架，并提供其它功能。
    缺点是self.dataset中的变量和方法都是不确定的，无法类型检查
    """
    def __init__(self):
        self.framework = None
        self.dataset = None

class TorchDataSet(TowheeDataSet):
    """
    pytorch的dataset
    """
    def __init__(self, torchDataSet: Dataset = None):
        super().__init__()
        self.framework = 'torch'
        self.dataset = torchDataSet

    def __len__(self):
        return len(self.dataset)

    def get_framework(self):
        return str(self.framework)
# class TFDataSet():
#
# d = TorchDataSet(selfdataset)
def get_dataset(name, **kwargs) -> TorchDataSet:
    """
    通过名字去找对应的dataset，
    提供一组默认的参数，用户也可以使用指定参数去覆盖
    """
    dataset_construct_map = {
        'mnist': datasets.MNIST,
        'cifar10': datasets.cifar.CIFAR10
        # 'imdb': IMDB  # ,()
    }
    default_args = {'root': 'data'}
    kwargs = {**default_args, **kwargs}
    torch_dataset = dataset_construct_map[name](
        **kwargs)  # , transform=transform, target_transform=target_transform, **kwargs)
    return TorchDataSet(torch_dataset)
