"""Dataset + transforms.

TEAMMATE B delivers: data/processed/vision_labels.csv
    columns -> filepath, skin_type, acne, dark_spots, redness, large_pores, wrinkles, dryness
Anna's training code only ever reads that CSV, so B can work fully in parallel.
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from skincare.config import CONCERNS, IMAGE_SIZE, SKIN_TYPES


def build_transforms(train: bool):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), norm])


class SkinDataset(Dataset):
    def __init__(self, csv_path, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB")
        y_type = torch.tensor(SKIN_TYPES.index(row["skin_type"]))
        y_concern = torch.tensor([float(row.get(c, 0)) for c in CONCERNS])
        return self.tf(img), y_type, y_concern


def make_loaders(train_csv, val_csv, batch_size=32, num_workers=2):
    return (
        DataLoader(SkinDataset(train_csv, True), batch_size=batch_size,
                   shuffle=True, num_workers=num_workers),
        DataLoader(SkinDataset(val_csv, False), batch_size=batch_size,
                   shuffle=False, num_workers=num_workers),
    )
