from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class ClassificationDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir=None,
        transform=None,
        minority_transform=None,
        minority_label=0,
    ):
        """
        Initializes the ClassificationDataset.

        :param root_dir: Directory with all the images, defaults to None
        :param transform: Transformation to apply to each image, defaults to None
        :param minority_transform: Special transformation for minority class images, defaults to None
        :param minority_label: Label indicating the minority class, defaults to 0
        :param df: DataFrame containing image file paths and labels, defaults to None
        """
        # Load the DataFrame from a CSV file if not provided
        self.df = df
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_label = minority_label

    def __getitem__(self, idx):
        # Get the image path and label for the given index
        image_path = self.df["filepath"].iloc[idx]
        image = Image.open(image_path)
        label = int(self.df["label"].iloc[idx])

        # Convert image to numpy array
        image = np.array(image)

        # Apply special transformation if the image belongs to the minority class
        if label == self.minority_label and self.minority_transform:
            image = self.minority_transform(image=image)["image"]
        # Otherwise, apply the general transformation if available
        elif self.transform:
            image = self.transform(image=image)["image"]

        # Return the transformed image and its label
        return image.astype(np.float32), label

    def __len__(self):
        # Return the total number of samples
        return len(self.df)
