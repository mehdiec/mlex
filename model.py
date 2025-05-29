import timm
import torch.nn as nn


def classification_head(in_features, n_classes, depth=2, hidden_dim=256, dropout=0.0):
    """Creates a classification head with specified depth and dropout.

    :param in_features: Number of input features to the classification head.
    :param n_classes: Number of output classes for classification.
    :param depth: Number of layers in the classification head, defaults to 2.
    :param hidden_dim: Number of hidden units in each hidden layer, defaults to 256.
    :param dropout: Dropout rate to apply after each hidden layer, defaults to 0.0.
    :return: A sequential model representing the classification head.
    """
    layers = []
    current_dim = in_features

    # Add hidden layers
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
        )
        current_dim = hidden_dim

    # Add final classification layer
    layers.extend([nn.Linear(current_dim, n_classes)])

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    """A module consisting of two convolutional layers each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """A module that applies max pooling followed by a DoubleConv operation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    """A module that applies a 1x1 convolution to reduce the number of channels."""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """An encoder module that processes input through a series of convolutional layers and a classification head."""

    def __init__(self, n_channels, nbase, n_classes, fc=None):
        super(Encoder, self).__init__()
        nbase = [(i + 1) * nbase for i in range(4)]
        self.inc = DoubleConv(n_channels, nbase[0])
        self.down1 = Down(nbase[0], nbase[1])
        self.down2 = Down(nbase[1], nbase[2])
        self.down3 = Down(nbase[2], nbase[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc if fc is not None else classification_head(nbase[3], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.avgpool(x4)
        x5 = x5.view(x5.size(0), -1)
        x5 = self.fc(x5)
        return x5


def build_model(
    model_name: str,
    num_classes: int,
    freeze: bool = False,
    pretrained: bool = True,
    dim_base: int = 32,
    classification_head_depth: int = 2,
    classification_head_dropout: float = 0.0,
):
    """Builds a model based on the specified architecture name.

    :param model_name: Name of the model architecture to build ('cnn_encoder', 'resnet', or 'efficientnet').
    :param num_classes: Number of output classes for the model.
    :param freeze: Whether to freeze the feature extractor layers, defaults to False.
    :param pretrained: Whether to use a pretrained model, defaults to True.
    :param dim_base: Base dimension for the encoder, defaults to 32.
    :param classification_head_depth: Depth of the classification head, defaults to 2.
    :param classification_head_dropout: Dropout rate for the classification head, defaults to 0.0.
    :return: A PyTorch model instance.
    """

    model = None

    if model_name == "cnn_encoder":
        model = Encoder(3, dim_base, num_classes)

    elif model_name == "resnet":
        model = timm.create_model("resnet18", pretrained=pretrained)
        infeat = model.fc.in_features

        # Freeze the feature extractor layers if specified
        if freeze:
            for name, param in model.named_parameters():
                if "fc" not in name:  # Don't freeze the final FC layer
                    param.requires_grad = False

        # Create and attach new classification head
        fc = classification_head(
            infeat,
            num_classes,
            classification_head_depth,
            infeat // 2,
            classification_head_dropout,
        )
        model.fc = fc

    elif model_name == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=pretrained)
        infeat = model.classifier.in_features

        # Freeze the feature extractor layers if specified
        if freeze:
            for name, param in model.named_parameters():
                if "classifier" not in name:  # Don't freeze the classifier
                    param.requires_grad = False

        # Create and attach new classification head
        fc = classification_head(
            infeat,
            num_classes,
            classification_head_depth,
            infeat // 2,
            classification_head_dropout,
        )
        model.classifier = fc

    return model
