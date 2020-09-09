from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import parameter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def data_split():
    root_dir = Path(parameter.data_path)
    x = []
    y = []
    classes_num = 0
    for classes in os.listdir(root_dir):
        for img_file in os.listdir(root_dir / classes):
            x.append(root_dir / classes / img_file)
            y.append(classes_num)
        classes_num += 1
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    return x_train, x_val, y_train, y_val


class TrainDataset(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x = x_train
        self.y = y_train
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]


class ValDataset(Dataset):
    def __init__(self, x_val, y_val, transform=None):
        self.x = x_val
        self.y = y_val
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]
