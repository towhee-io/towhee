import os

import pandas as pd
from pandas import Series
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    """
        PytorchImageDataset is a dataset class for PyTorch.

        Args:
            image_path (:obj:`str`):
                Path to the images for your dataset.

            label_file (:obj:`str`):
                 Path to your label file. The label file should be a csv file. The columns in this file should be [image_name, category],
                'image_name' is the path of your images, 'category' is the label of accordance image. For example:
                [image_name, dog] for one row. Note that the first row should be[image_name, category]

            data_transform (:obj:`Compose`):
                PyTorch transform of the input images.
        """

    def __init__(self, image_path, label_file, data_transform=None):
        self.image_path = image_path
        self.label_file = label_file
        self.data_transform = data_transform

        df = pd.read_csv(self.label_file)
        image_names = Series.to_numpy(df['image_name'])
        images = [i + ".jpg" for i in image_names]
        self.images = [os.path.join(self.image_path, i) for i in images]

        categories = Series.to_numpy(df['category'])
        # Count the categories
        breed_set = set(categories)
        breed_list = list(breed_set)
        dic = dict()
        for i in range(len(breed_list)):
            dic[breed_list[i]] = i
        self.labels = [dic[categories[i]] for i in range(len(categories))]

    def __getitem__(self, index):
        label = self.labels[index]
        fn = self.images[index]
        img = Image.open(fn)
        if self.data_transform:
            img = self.data_transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)
