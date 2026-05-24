import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class HotnessDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Load the pre-processed CSV file
        self.annotations: pd.DataFrame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Read exact filename from the 'Filename' column
        img_name = str(self.annotations['Filename'].iloc[index])
        img_path = os.path.join(self.img_dir, img_name)

        # Load image and ensure it's RGB
        image = Image.open(img_path).convert("RGB")

        # Read continuous rating from the 'Rating' column
        rating = float(self.annotations['Rating'].iloc[index])
        y_label = torch.tensor(rating, dtype=torch.float32)

        # Apply image augmentations/normalizations
        if self.transform:
            image = self.transform(image)

        return image, y_label