import argparse
import itertools
import logging
import pathlib
import random
import time

from PIL import Image
import pydantic
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms

from . import custom_model
from ...ttt import board_draw

from typing import Callable, Iterator

# Size to which the image will be resized to.
_IMAGE_SIZE = 224

_EPOCHS = 20
_BATCHES_PER_EPOCH = 50
_BATCH_SIZE = 32

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saved at the end of every epoch.
_CHECKPOINT_FILE = pathlib.Path("_checkpoint.pth")


# Custom data saved with checkpoint.
class _EpochStats(pydantic.BaseModel):
    duration: float
    loss: float


_EpochStatsList = pydantic.RootModel[list[_EpochStats]]


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
            label_tensor = torch.zeros((custom_model.NUM_CLASSES, 3))
            for i, c in enumerate(board_array):
                label_tensor[i, char_to_class[c]] = 1
            image_tensor: torch.Tensor = self._transform(image)
            yield image_tensor.to(_DEVICE), label_tensor.to(_DEVICE)


def _save_checkpoint(
    fname: pathlib.Path,
    model: custom_model.TicTacToeVision,
    optimizer: optim.Optimizer,
    epoch_stats: _EpochStatsList,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_stats": epoch_stats.model_dump(),
        },
        fname,
    )
    logging.info(f"Saved checkpoint to {fname}")


def _load_checkpoint(
    fname: pathlib.Path,
    model: custom_model.TicTacToeVision,
    optimizer: optim.Optimizer,
    epoch_stats: _EpochStatsList,
) -> None:
    """Returns training time."""
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_stats.model_validate(checkpoint["epoch_stats"])
    logging.info(f"Loaded checkpoint from {fname}")


def _train(use_checkpoints: bool):
    logging.info(f"Using device: {_DEVICE}")

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

    model = custom_model.TicTacToeVision().to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {params:_} parameters.")
    size_in_mb = params * 4 / 1024 / 1024
    print(f"Model size: {size_in_mb:.2f} MB")

    epoch_stats = _EpochStatsList([])

    if not use_checkpoints:
        logging.info("Not using checkpoints.")
    else:
        if _CHECKPOINT_FILE.exists():
            logging.info(f"Checkpoint exists. Loading.")
            start_epoch = _load_checkpoint(
                fname=_CHECKPOINT_FILE,
                model=model,
                optimizer=optimizer,
                epoch_stats=epoch_stats,
            )
        else:
            logging.info(f"Checkpoint does not exist. Starting from scratch.")

    criterion = nn.CrossEntropyLoss()

    # Training loop.
    start_epoch = len(epoch_stats.root)
    for epoch in range(start_epoch, _EPOCHS):  # number of epochs
        model.train()
        running_loss = 0.0
        start_time = time.time()
        # Normally whole dataset is one epoch.
        # However, we have infinite data. So we arbitrarily define epoch size.
        for index, (images, labels) in enumerate(
            itertools.islice(train_loader, _BATCHES_PER_EPOCH)
        ):
            optimizer.zero_grad()
            outputs = model(images)
            # Convert this into view of [batch * 9, 3], i.e. (batch * 9) independent classes.
            # Note: We could transpose to keep dim1 as the classes. But it's simpler to just have 2 dimensions.
            loss = criterion(outputs.view(-1, 3), labels.view(-1, 3))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"Epoch: {epoch}/{_EPOCHS}, Index: {index}/{_BATCHES_PER_EPOCH}, Avg. Loss this epoch: {running_loss/(index + 1)}"
            )

        epoch_time = time.time() - start_time

        epoch_stats.root.append(
            _EpochStats(duration=epoch_time, loss=running_loss / _BATCHES_PER_EPOCH)
        )

        model.eval()
        # TODO: Can put out-of-sample evaluation code here.
        # For now, just show one result.
        image, label = next(iter(test_dataset))
        print(f"True Label: {label}")
        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
        print(f"Inferred Label: {outputs}")
        if use_checkpoints:
            _save_checkpoint(
                fname=_CHECKPOINT_FILE,
                model=model,
                optimizer=optimizer,
                epoch_stats=epoch_stats,
            )

    model.save_savetensors()


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
