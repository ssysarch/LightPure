import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class GTSRB(Dataset):

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = "Train" if train else "Test"
        self.csv_file_name = "Train.csv" if train else "Test.csv"

        csv_file_path = os.path.join(
            root_dir,  self.csv_file_name
        )
        print(root_dir)
        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):

        # print(self.root_dir)
        # print(self.csv_data.iloc[idx, 7])
        img_path = os.path.join(
            self.root_dir,
            self.csv_data.iloc[idx, 7],
        )
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 6]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ]
)


