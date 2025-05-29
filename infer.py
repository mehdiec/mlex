import configargparse
import torch
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml
from torch.utils.data import DataLoader, Dataset

from model import build_model
from pl_module import ClassificationPLModule


def parse_infer_args():
    parser = configargparse.ArgumentParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="Config file path",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.txt",
        help="Output file path for predictions",
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--val_threshold",
        type=float,
        default=0.7071,
        help="Validation threshold.",
    )
    parser.add_argument(
        "--get_proba",
        type=bool,
        default=False,
        help="Get proba.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet",
        help="Model name.",
    )

    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image


def main(args):
    # Load config
    config = load_config(args.config) if args.config else {}

    # Get configuration parameters
    device_index = args.device_index or config.get("device_index", 0)
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or config.get("batch_size", 32)
    get_proba = args.get_proba or config.get("get_proba", False)
    val_threshold = args.val_threshold or config.get("val_threshold", 0.7071)
    # Build model
    model = build_model(
        model_name=config.get("model_type", "resnet"),
        num_classes=config.get("num_classes", 1),
        freeze=False,
        pretrained=False,
        classification_head_depth=config.get("classification_head_depth", 2),
        classification_head_dropout=0.0,
    )

    # Load model from checkpoint
    model_pl = ClassificationPLModule.load_from_checkpoint(
        args.checkpoint_path, model=model, strict=False
    )
    model_pl.to(device)
    model_pl.eval()

    # Get sorted list of image files
    input_dir = Path(args.input_dir)
    image_files = sorted(
        [f for f in input_dir.glob("*.jpg")], key=lambda x: int(x.stem)
    )

    # Create dataset and dataloader
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageDataset(image_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    predictions = []
    probabilities = []
    # Process images in batches
    with torch.no_grad():
        for batch_images in tqdm(dataloader, desc="Processing images"):
            batch_tensor = batch_images.to(device).float()

            # Get predictions
            outputs = model_pl(batch_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Apply threshold
            batch_preds = (probs >= val_threshold).astype(int)
            predictions.extend(batch_preds.flatten())
            if get_proba:
                probabilities.extend(probs.flatten())

    # Save predictions
    output_file = Path(args.output_file)
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    if get_proba:
        with open(output_file.with_suffix(".proba"), "w") as f:
            for proba in probabilities:
                f.write(f"{proba}\n")

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    args = parse_infer_args()
    main(args)
