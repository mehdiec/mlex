import os

import configargparse
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from callbacks import MetricsCallback
from data_module import ClassificationDataModule
from model import build_model
from pl_module import ClassificationPLModule
from utils import save_config_and_checkpoints
import albumentations as A


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def parse_train_args():
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
        "--device_index", type=int, default=0, help="CUDA device index."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=0.00004, help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        help="Scheduler type (e.g., 'cosine', 'reduce_on_plateau').",
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of input channels."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/mnt/DATA/checkpoints_cp_lat",
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/Volumes/u934/equipe_bellaiche/m_ech-chouini/TA_bench/event_only/mlex/ml_exercise_therapanacea/train_img",
        help="Path to the training data set.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resunet",
        help="type of model",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=0.2,
        help="weight for the positive class",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="number of classes",
    )
    parser.add_argument(
        "--freeze",
        type=bool,
        default=False,
        help="freeze the model",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="use pretrained model",
    )
    parser.add_argument(
        "--classification_head_depth",
        type=int,
        default=2,
        help="depth of the classification head",
    )
    parser.add_argument(
        "--classification_head_dropout",
        type=float,
        default=0.5,
        help="dropout of the classification head",
    )
    parser.add_argument(
        "--dim_base",
        type=int,
        default=32,
        help="base dimension of the model",
    )

    return parser.parse_args()


def main(args):
    # Load config from YAML if provided
    config = load_config(args.config) if args.config else {}

    # Use arguments from parser, falling back to config if available
    device_index = args.device_index or config.get("device_index", 0)
    num_epochs = args.num_epochs or config.get("num_epochs", 1000)
    lr = args.lr or config.get("lr", 0.00004)
    weight_decay = args.weight_decay or config.get("weight_decay", 0.0001)
    scheduler = args.scheduler or config.get("scheduler", "cosine")
    checkpoint_dir = args.checkpoint_dir or config.get(
        "checkpoint_dir", "/mnt/DATA/checkpoints_gender"
    )
    root_dir = args.root_dir or config.get("root_dir")
    batch_size = args.batch_size or config.get("val_path")
    model_type = args.model_type or config.get("model_type", "resunet")
    pos_weight = args.pos_weight or config.get("pos_weight", 0.2)
    num_classes = args.num_classes or config.get("num_classes", 1)
    freeze = args.freeze or config.get("freeze", False)
    pretrained = args.pretrained or config.get("pretrained", True)
    classification_head_depth = args.classification_head_depth or config.get(
        "classification_head_depth", 2
    )
    classification_head_dropout = args.classification_head_dropout or config.get(
        "classification_head_dropout", 0.5
    )
    dim_base = args.dim_base or config.get("dim_base", 32)

    model = build_model(
        model_type,
        num_classes=num_classes,
        freeze=freeze,
        pretrained=pretrained,
        classification_head_depth=classification_head_depth,
        classification_head_dropout=classification_head_dropout,
        dim_base=dim_base,
    )

    model.to(f"cuda:{device_index}")
    comet_logger = CometLogger(api_key=os.environ["COMET_API_KEY"], project="ml-ex")
    comet_logger.log_hyperparams(config)

    dirpath = save_config_and_checkpoints(config, checkpoint_dir, model_type)

    # Define transforms for minority class (more aggressive augmentation)
    minority_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.RandomRotate90(p=0.7),
            A.Rotate(p=0.7),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=150, sigma=150 * 0.05, alpha_affine=150 * 0.03, p=0.7
                    ),
                    A.GridDistortion(p=0.7),
                    A.OpticalDistortion(distort_limit=1.5, shift_limit=0.7, p=0.7),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=0.7
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.7),
                    A.MotionBlur(blur_limit=3, p=0.7),
                    A.MedianBlur(blur_limit=3, p=0.7),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2.0, p=0.7),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.7),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.7),
                ],
                p=0.3,
            ),
        ]
    )

    # Define transforms for majority class (lighter augmentation)
    val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.RandomRotate90(p=0.3),
            A.Rotate(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )

    # Initialize DataModule
    data_module = ClassificationDataModule(
        root_dir,
        transform=transform,
        minority_transform=minority_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        num_workers=12,
        val_split=0.2,
        seed=55,
    )

    model_pl = ClassificationPLModule(
        model,
        lr=lr,
        epoch=num_epochs,
        weight_decay=weight_decay,
        scheduler=scheduler,
        pos_weight=pos_weight,
    )

    # Logger and trainer setup
    trainer = pl.Trainer(
        logger=comet_logger,
        max_epochs=num_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=dirpath,
                save_top_k=1,
                monitor="val_HTER",
            ),
            EarlyStopping(
                monitor="val_HTER",
                min_delta=0.001,
                patience=10,
                verbose=True,
                mode="min",
            ),
            MetricsCallback(),
        ],
        devices=[device_index],
    )

    trainer.fit(model_pl, datamodule=data_module)


if __name__ == "__main__":
    args = parse_train_args()
    main(args)
