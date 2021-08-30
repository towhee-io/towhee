from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
        PytorchImageDataset is a dataset class for PyTorch.

        Args:
            image_path (:obj:`str`):
                Path to the images for your dataset.

            label_file (:obj:`str`):
                 Path to your label file. The label file should be a csv file. The columns in this file should be [image_name, category],
                'image_name' is the path of your images, 'category' is the label of accordance image. For example:
                [image_name.jpg, dog] for one row. Note that the first row should be[image_name, category]

            data_transform (:obj:`Compose`):
                PyTorch transform of the input images.
        """

    def __init__(self, image_path, label_file, data_transform=None):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
