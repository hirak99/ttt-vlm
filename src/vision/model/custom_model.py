import abc
import logging
import pathlib

from PIL import Image
import safetensors.torch
import torch
from torch import nn
import torchvision.transforms as transforms

from typing import Callable, override

# This is the number of cells.
# True constant - this will not change.
NUM_CLASSES = 9

# Size to which the image will be resized to.
_IMAGE_SIZE = 224

CHAR_TO_CLASSID = {
    "X": 0,
    "O": 1,
    ".": 2,
}
_CLASSID_TO_CHAR = {v: k for k, v in CHAR_TO_CLASSID.items()}


class BaseModel(nn.Module, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def file_suffix(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def image_to_input(cls, image: Image.Image) -> torch.Tensor:
        pass

    def recognize(self, image: Image.Image) -> list[str]:
        image_input = self.image_to_input(image)
        with torch.no_grad():
            logits = self(image_input.unsqueeze(0))

        # Argmax along the last dimension.
        class_ids = logits.argmax(dim=-1)
        # We have only one image in this batch.
        assert class_ids.shape[0] == 1

        class_ids_np = class_ids.cpu().numpy()
        return [_CLASSID_TO_CHAR[class_id] for class_id in class_ids_np[0]]

    def save_savetensors(self):
        fname = pathlib.Path("_data") / f"model_{self.file_suffix()}.safetensor"
        safetensors.torch.save_file(self.state_dict(), fname)
        logging.info(f"Saved model to {fname}")

    def load_safetensors(self):
        fname = pathlib.Path("_data") / f"model_{self.file_suffix()}.safetensor"
        self.load_state_dict(safetensors.torch.load_file(fname))
        logging.info(f"Loaded model from {fname}")


class CnnV1(BaseModel):
    _transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
        [
            transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
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
            nn.Linear(512, NUM_CLASSES * 3),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, NUM_CLASSES, 3)
        return x

    @override
    @classmethod
    def file_suffix(cls) -> str:
        return "cnnv1"

    @override
    @classmethod
    def image_to_input(cls, image: Image.Image) -> torch.Tensor:
        return cls._transform(image)
