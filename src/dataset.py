"""
Dataset PyTorch pour APTOS 2019 + creation des splits train/val/test.
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from . import config
from .preprocess import load_and_preprocess


class APTOSDataset(Dataset):
    """Chaque element = (image pretraitee en tenseur, label 0-4)."""

    def __init__(self, df, transform=None, use_clahe=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = config.TRAIN_IMAGES / f"{row['id_code']}.png"

        # Pretraitement OpenCV -> numpy uint8 (H, W, 3)
        img = load_and_preprocess(img_path, size=config.IMG_SIZE,
                                  use_clahe=self.use_clahe)
        img = Image.fromarray(img)              # -> PIL pour torchvision

        if self.transform:
            img = self.transform(img)

        label = int(row["diagnosis"])
        return img, label


# ---------------------------------------------------------------------------
# Transformations (augmentation a l'entrainement, simple a la validation)
# ---------------------------------------------------------------------------
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Splits stratifies (on garde la meme repartition des stades partout)
# ---------------------------------------------------------------------------
def make_splits():
    df = pd.read_csv(config.TRAIN_CSV)

    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_SIZE + config.TEST_SIZE,
        stratify=df["diagnosis"], random_state=config.SEED,
    )
    rel_test = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["diagnosis"], random_state=config.SEED,
    )
    return train_df, val_df, test_df


def get_loaders():
    train_df, val_df, test_df = make_splits()
    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    train_ds = APTOSDataset(train_df, get_transforms(train=True))
    val_ds = APTOSDataset(val_df, get_transforms(train=False))
    test_ds = APTOSDataset(test_df, get_transforms(train=False))

    # num_workers=0 sous Windows pour eviter les soucis de multiprocessing
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def compute_class_weights():
    """Poids inversement proportionnels a la frequence des classes
    (APTOS est desequilibre : beaucoup de stade 0)."""
    import torch
    df = pd.read_csv(config.TRAIN_CSV)
    counts = df["diagnosis"].value_counts().sort_index().values
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)
