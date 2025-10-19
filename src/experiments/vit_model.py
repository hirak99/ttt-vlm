import argparse
import itertools
import logging
import pathlib
import random

from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms

from ..ttt import board_draw

from typing import Callable, Iterator

# This is the number of cells.
# True constant - this will not change.
_NUM_CLASSES = 9

# Size to which the image will be resized to.
_IMAGE_SIZE = 224

_EPOCHS = 100
_BATCHES_PER_EPOCH = 50
_BATCH_SIZE = 32

_DEVICE = "cuda"

_CHECKPOINT_FILE = pathlib.Path("_checkpoint.pth")


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


class _TicTacToeViT(nn.Module):
    def __init__(self):
        super(_TicTacToeViT, self).__init__()

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
            nn.Linear(512, _NUM_CLASSES * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, _NUM_CLASSES, 3)
        return x


def _save_checkpoint(
    fname: pathlib.Path, model: _TicTacToeViT, optimizer: optim.Optimizer, epoch: int
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        fname,
    )
    logging.info(f"Saved checkpoint to {fname}")


def _load_checkpoint(
    fname: pathlib.Path, model: _TicTacToeViT, optimizer: optim.Optimizer
) -> int:
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info(f"Loaded checkpoint from {fname}")
    return checkpoint["epoch"]


def _train(use_checkpoints: bool):
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

    model = _TicTacToeViT().to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {params:_} parameters.")
    size_in_mb = params * 4 / 1024 / 1024
    print(f"Model size: {size_in_mb:.2f} MB")

    start_epoch = 0
    if not use_checkpoints:
        logging.info("Not using checkpoints.")
    else:
        if _CHECKPOINT_FILE.exists():
            logging.info(f"Checkpoint exists. Loading.")
            start_epoch = _load_checkpoint(_CHECKPOINT_FILE, model, optimizer)
        else:
            logging.info(f"Checkpoint does not exist. Starting from scratch.")

    criterion = nn.CrossEntropyLoss()

    # Training loop.
    for epoch in range(start_epoch, _EPOCHS):  # number of epochs
        model.train()
        running_loss = 0.0
        # Normally whole dataset is one epoch.
        # However, we have infinite data. So we arbitrarily define epoch size.
        for index, (images, labels) in enumerate(
            itertools.islice(train_loader, _BATCHES_PER_EPOCH)
        ):
            optimizer.zero_grad()
            outputs = model(images)
            # Convert this into view of [batch * 9, 3], i.e. batch * 9 independent classes.
            loss = criterion(outputs.view(-1, 3), labels.view(-1, 3))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"Epoch: {epoch}, Index: {index}/{_BATCHES_PER_EPOCH}, Loss: {running_loss/(index + 1)}"
            )

        model.eval()
        # TODO: Can put out-of-sample evaluation code here.
        image, label = next(iter(test_dataset))
        print(f"True Label: {label}")
        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
        print(f"Inferred Label: {outputs}")
        if use_checkpoints:
            _save_checkpoint(_CHECKPOINT_FILE, model, optimizer, epoch)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        default=False,
        help="Do not use checkpoints.",
    )
    args = parser.parse_args()

    _train(use_checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
