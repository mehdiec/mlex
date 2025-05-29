from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import pytorch_lightning as pl
from data_loader import ClassificationDataset
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def make_balanced_sampler(labels_np):
    """
    Creates a WeightedRandomSampler to balance the dataset by assigning
    higher weights to minority class samples.

    :param labels_np: Numpy array of labels for the dataset.
    :return: A WeightedRandomSampler object.
    """
    class_count = np.bincount(labels_np)
    class_weights = 1.0 / class_count  # inversely proportional
    sample_weights = class_weights[labels_np]  # each index weight
    return WeightedRandomSampler(
        sample_weights, num_samples=len(labels_np), replacement=True
    )


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        transform=None,
        val_transform=None,
        minority_transform=None,
        batch_size=16,
        num_workers=4,
        val_split=0.2,
        seed=42,
    ):
        """
        Initializes the ClassificationDataModule.

        :param root_dir: Directory containing the dataset.
        :param transform: Transformation to apply to training images, defaults to None.
        :param val_transform: Transformation to apply to validation and test images, defaults to None.
        :param minority_transform: Special transformation for minority class images, defaults to None.
        :param batch_size: Number of samples per batch, defaults to 16.
        :param num_workers: Number of subprocesses to use for data loading, defaults to 4.
        :param val_split: Fraction of training data to use for validation, defaults to 0.2.
        :param seed: Random seed for reproducibility, defaults to 42.
        """
        super().__init__()
        self.csv_file = Path(root_dir) / "label_train.txt"
        self.root_dir = Path(root_dir) / "train_img"
        self.transform = transform
        self.val_transform = val_transform
        self.minority_transform = minority_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.csv_file, header=None, names=["label"])
        labels = df.iloc[:, 0].to_numpy()
        df["filepath"] = df.index.map(lambda i: self.root_dir / f"{(i + 1):06d}.jpg")
        df["filename"] = df.index.map(lambda i: f"{(i + 1):06d}.jpg")

        # Stratified split to ensure class distribution is maintained
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=0.10,
            stratify=labels,
            random_state=42,
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_split,
            stratify=labels[train_val_idx],
            random_state=self.seed,
        )

        self.df_train = df.iloc[train_idx].reset_index(drop=True)
        self.df_val = df.iloc[val_idx].reset_index(drop=True)
        self.df_test = df.iloc[test_idx].reset_index(drop=True)

        # Build datasets
        self.dataset_train = ClassificationDataset(
            root_dir=self.root_dir,
            transform=self.transform,
            minority_transform=self.minority_transform,
            minority_label=0,
            df=self.df_train,
        )

        self.dataset_val = ClassificationDataset(
            root_dir=self.root_dir,
            df=self.df_val,
            transform=self.val_transform,
        )

        self.dataset_test = ClassificationDataset(
            root_dir=self.root_dir,
            df=self.df_test,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        labels_np = self.df_train.iloc[:, 0].to_numpy()
        sampler = make_balanced_sampler(labels_np)

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
