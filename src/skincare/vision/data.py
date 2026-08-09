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
        # -1 means this image was never annotated for that concern, which is common
        # once several source datasets are merged. It must not be recorded as 0:
        # "not annotated" is not the same claim as "confirmed absent", and treating
        # it that way teaches the model to predict negative on under-annotated
        # concerns. Both the loss and the metrics mask these positions out.
        vals = []
        for c in CONCERNS:
            v = row.get(c)
            vals.append(float(v) if pd.notna(v) else -1.0)
        y_concern = torch.tensor(vals, dtype=torch.float32)
        return self.tf(img), y_type, y_concern


def make_loaders(train_csv, val_csv, batch_size=32, num_workers=2):
    return (
        DataLoader(SkinDataset(train_csv, True), batch_size=batch_size,
                   shuffle=True, num_workers=num_workers),
        DataLoader(SkinDataset(val_csv, False), batch_size=batch_size,
                   shuffle=False, num_workers=num_workers),
    )
