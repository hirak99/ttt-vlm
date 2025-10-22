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

from . import base_model
from . import registry
from ...ttt import board_draw

from typing import Iterator

_EPOCHS = 100
_BATCHES_PER_EPOCH = 50
_BATCH_SIZE = 32

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom data saved with checkpoint.
class _EpochStats(pydantic.BaseModel):
    duration: float
    loss: float


_EpochStatsList = pydantic.RootModel[list[_EpochStats]]


class _IterableData(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, model_class: base_model.BaseModel):
        super().__init__()
        self._model_class = model_class.__class__

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            # Note: For vision training, we do not care at the moment if the position is valid.
            board_array = [random.choice(["X", "O", "."]) for _ in range(9)]
            render_params = board_draw.RenderParams.random()
            image: Image.Image = board_draw.to_image(board_array, render_params)
            # NOTE: We are labeling as integers denoting class-ids. On this,
            # one-hot computation will be done implicitly during loss
            # computation.
            # We could compute one-hot if we wanted and also passed that instead -
            # onehot = F.one_hot(class_ids, 3)
            class_ids = torch.tensor(
                [base_model.CHAR_TO_CLASSID[c] for c in board_array]
            )
            image_input = self._model_class.image_to_input(image)
            yield image_input.to(_DEVICE), class_ids.to(_DEVICE)


# Possible definition of dataset without iteration. This may help to speed up
# training by reducing CPU work per cycle.
#
# class _InMemoryData(Dataset[tuple[torch.Tensor, torch.Tensor]]):
#     def __init__(self, model_class: base_model.BaseModel, length: int) -> None:
#         super().__init__()
#         self._iterable = iter(_IterableData(model_class))
#         self._data = list[tuple[torch.Tensor, torch.Tensor]](self._iterable)
#         self._length = length
#
#     def __len__(self) -> int:
#         return self._length
#
#     def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         while len(self._data) < index + 1:
#             self._data.append(next(self._iterable))
#         return self._data[index]


class _Trainer:
    def __init__(self, model: base_model.BaseModel):
        self._model = model.to(_DEVICE)

        self._checkpointfile = (
            pathlib.Path("_data") / f"_checkpoint_{self._model.file_suffix}.pth"
        )

    def _save_checkpoint(
        self,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        epoch_stats: _EpochStatsList,
    ):
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch_stats": epoch_stats.model_dump(),
            },
            self._checkpointfile,
        )
        logging.info(f"Saved checkpoint to {self._checkpointfile}")

    def _load_checkpoint(
        self,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        epoch_stats: _EpochStatsList,
    ) -> None:
        """Returns training time."""
        checkpoint = torch.load(self._checkpointfile)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Some earlier models were trained without a scheduler.
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_stats.root = _EpochStatsList.model_validate(
            checkpoint["epoch_stats"]
        ).root
        logging.info(f"Loaded checkpoint from {self._checkpointfile}")
        for index, epoch_stat in enumerate(epoch_stats.root):
            logging.info(f"Loss at epoch {index}: {epoch_stat.loss}")

    def train(self, use_checkpoints: bool):
        logging.info(f"Using device: {_DEVICE}")

        train_dataset = _IterableData(self._model)
        train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE)

        test_dataset = _IterableData(self._model)

        optimizer = optim.Adam(self._model.parameters(), lr=1e-4)

        # Initialize the Scheduler
        # 'mode'='min' means it monitors a loss (e.g., validation loss) and reduces LR if it stops decreasing.
        # 'factor'=0.5 means it will cut the LR by half (e.g., 1e-4 -> 5e-5).
        # 'patience'=15 means it will wait 15 epochs without improvement before reducing the LR.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=15,
        )

        params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"Model has {params:_} parameters.")
        size_in_mb = params * 4 / 1024 / 1024
        print(f"Model size: {size_in_mb:.2f} MB")

        epoch_stats = _EpochStatsList([])

        if not use_checkpoints:
            logging.info("Not using checkpoints.")
        else:
            if self._checkpointfile.exists():
                logging.info(f"Checkpoint ({self._checkpointfile!r}) exists. Loading.")
                start_epoch = self._load_checkpoint(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_stats=epoch_stats,
                )
            else:
                logging.info(
                    f"Checkpoint ({self._checkpointfile!r}) does not exist. Starting from scratch."
                )

        criterion = nn.CrossEntropyLoss()

        # Training loop.
        start_epoch = len(epoch_stats.root)
        for epoch in range(start_epoch, _EPOCHS):  # number of epochs
            self._model.train()
            running_loss = 0.0
            print(f"Current LR: {optimizer.param_groups[0]['lr']}")
            start_time = time.time()
            # Normally whole dataset is one epoch.
            # However, we have infinite data. So we arbitrarily define epoch size.
            for index, (images, labels) in enumerate(
                itertools.islice(train_loader, _BATCHES_PER_EPOCH)
            ):
                optimizer.zero_grad()
                logits = self._model(images)
                # Convert logits into view of [batch * 9, 3], i.e. (batch * 9) independent classes.
                # Note: We could transpose to keep dim1 as the classes. But it's simpler to just have 2 dimensions.
                # Also, multi-target is ambiguous for class-id.
                loss = criterion(logits.view(-1, 3), labels.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print(
                    f"Epoch: {epoch}/{_EPOCHS}, Index: {index}/{_BATCHES_PER_EPOCH}, Avg. Loss this epoch: {running_loss/(index + 1)}"
                )

            scheduler.step(running_loss)

            epoch_time = time.time() - start_time

            epoch_stats.root.append(
                _EpochStats(duration=epoch_time, loss=running_loss / _BATCHES_PER_EPOCH)
            )

            self._model.eval()
            # TODO: Can put out-of-sample evaluation code here.
            # For now, just show one result.
            image_input, label = next(iter(test_dataset))
            print(f"True Label: {label}")
            with torch.no_grad():
                logits = self._model(image_input.unsqueeze(0))
            print(f"Inferred Label: {logits}")
            if use_checkpoints:
                self._save_checkpoint(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_stats=epoch_stats,
                )

        self._model.save_savetensor()


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

    model = registry.default_model()
    assert model is not None

    trainer = _Trainer(model)
    trainer.train(use_checkpoints=not args.no_checkpoints)


if __name__ == "__main__":
    main()
