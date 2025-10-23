"""Custom model definitions.

When you add model here, also add to the registry.

Do not instantiate directly, instead use the registry.
"""

from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms

from . import base_model

from typing import Callable, override, Type


class _CnnV1(base_model.BaseModel):
    _transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224 -> 224x224
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 14x14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, base_model.NUM_CLASSES * 3),
        )

    @override
    @classmethod
    def image_to_input(cls, image: Image.Image) -> torch.Tensor:
        return cls._transform(image)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, base_model.NUM_CLASSES, 3)
        return x


class _CnnV2(base_model.BaseModel):
    _transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # New Size = floor of:
            # (input_size + 2 * padding - kernel_size) / stride + 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 128x128 -> 128x128
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 16x16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, base_model.NUM_CLASSES * 3),
        )

    @override
    @classmethod
    def image_to_input(cls, image: Image.Image) -> torch.Tensor:
        return cls._transform(image)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, base_model.NUM_CLASSES, 3)
        return x


class _CnnV3(base_model.BaseModel):
    _transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # New Size = floor of:
            # (input_size + 2 * padding - kernel_size) / stride + 1
            #
            # Widened the channels sizes for improved capacity compared to CnnV2.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 128x128 -> 128x128
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Deepened it for improved learning.
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024),
            nn.ReLU(),
            # Regularization. Not super needed if we train on new data every batch.
            nn.Dropout(0.2),
            # Added another layer. I think this can be a good supporting change
            # and extract some non-linearly. This is a much lighter layer
            # compared to the first flatten.
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, base_model.NUM_CLASSES * 3),
        )

    @override
    @classmethod
    def image_to_input(cls, image: Image.Image) -> torch.Tensor:
        return cls._transform(image)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, base_model.NUM_CLASSES, 3)
        return x


def for_registry() -> dict[str, Type[base_model.BaseModel]]:
    return {
        "cnnv1": _CnnV1,
        "cnnv2": _CnnV2,
        "cnnv3": _CnnV3,
    }
