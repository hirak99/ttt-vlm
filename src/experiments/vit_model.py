import itertools
import logging
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from transformers import ViTConfig
from transformers import ViTModel

from ..ttt import board_draw

from typing import Callable, Iterator

# This is the number of cells.
# True constant - this will not change.
_NUM_CLASSES = 9

# Size to which the image will be resized to.
_IMAGE_SIZE = 224

_BATCH_SIZE = 32
_SAMPLES_PER_EPOCH = 20
_EPOCHS = 100

_DEVICE = "cuda"


class _TicTacToeDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, transform: transforms.Compose):
        self._transform: Callable[[Image.Image], torch.Tensor] = transform

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            # Note: For vision training, we do not care at the moment if the position is valid.
            board_array = [random.choice(["X", "O", "."]) for _ in range(9)]
            render_params = board_draw.RenderParams.random()
            image: Image.Image = board_draw.to_image(board_array, render_params)
            char_to_class = {
                "X": 0,
                "O": 1,
                ".": 2,
            }
            label_tensor = torch.zeros((_NUM_CLASSES, 3))
            for i, c in enumerate(board_array):
                label_tensor[i, char_to_class[c]] = 1
            image_tensor: torch.Tensor = self._transform(image)
            yield image_tensor.to(_DEVICE), label_tensor.to(_DEVICE)


# Configuration to instantiate ViTModel.
_CONFIG = ViTConfig(
    image_size=_IMAGE_SIZE,
)


class _TicTacToeViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super(_TicTacToeViT, self).__init__()
        self._vit = ViTModel(config)
        self._fc = nn.Linear(768, 3 * _NUM_CLASSES)

    def forward(self, x):
        x = self._vit(x).last_hidden_state
        print(x.shape)  # [32, <model-specific>, 768]
        x = x.mean(dim=1)  # Global average pooling across patches.
        print(x.shape)  # [32, 768]
        x = self._fc(x)
        print(x.shape)  # [32, 27]
        x = x.view(-1, _NUM_CLASSES, 3)  # Shape: [batch_size, 9, 3]
        return x


def _train():
    transform = transforms.Compose(
        [
            transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    # Example of how you'd train the model
    train_dataset = _TicTacToeDataset(transform)
    train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE)

    test_dataset = _TicTacToeDataset(transform)

    model = _TicTacToeViT(_CONFIG).to(_DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    # Training loop.
    for epoch in range(_EPOCHS):  # number of epochs
        model.train()
        running_loss = 0.0
        # Normally whole dataset is one epoch.
        # However, we have infinite data. So we arbitrarily define epoch size.
        for index, (images, labels) in enumerate(
            itertools.islice(train_loader, _SAMPLES_PER_EPOCH)
        ):
            optimizer.zero_grad()
            outputs = model(images)
            # Convert this into view of [batch * 9, 3], i.e. batch * 9 independent classes.
            loss = criterion(outputs.view(-1, 3), labels.view(-1, 3))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"Epoch: {epoch}, Index: {index}/{_SAMPLES_PER_EPOCH}, Loss: {running_loss/(index + 1)}"
            )

        model.eval()
        # TODO: Can put out-of-sample evaluation code here.
        image, label = next(iter(test_dataset))
        print(f"True Label: {label}")
        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
        print(f"Inferred Label: {outputs}")


def main():
    logging.basicConfig(level=logging.INFO)
    _train()


if __name__ == "__main__":
    main()
