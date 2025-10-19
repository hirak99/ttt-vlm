import itertools

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from transformers import ViTConfig
from transformers import ViTModel

# Size to which the image will be resized to.
_IMAGE_SIZE = 32

_NUM_CLASSES = 9

_BATCH_SIZE = 32
_SAMPLES_PER_EPOCH = 10


class TicTacToeDataset(IterableDataset[tuple[Image.Image, torch.Tensor]]):
    def __init__(self, transform: transforms.Compose | None = None):
        self._transform = transform

    def __iter__(self):
        while True:
            image, label = Image.new("RGB", (32, 32)), torch.tensor(
                [float(x) for x in [0, 0, 0, 0, 0, 0, 0, 0, 0]]
            )
            image = self._transform(image) if self._transform else image
            yield image, label


# Configuration to instantiate ViTModel.
_CONFIG = ViTConfig(
    image_size=_IMAGE_SIZE,
)


class _TicTacToeViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super(_TicTacToeViT, self).__init__()
        self._vit = ViTModel(config)
        self._fc = nn.Linear(768, _NUM_CLASSES)

    def forward(self, x):
        x = self._vit(x).last_hidden_state
        # print(x.shape)  # [32, 17, 768]
        x = x.mean(dim=1)  # Global average pooling across patches
        # print(x.shape)  # [32, 768]
        x = self._fc(x)  # Final classification
        # print(x.shape)
        return x


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    # Example of how you'd train the model
    train_dataset = TicTacToeDataset(transform)
    train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE)

    model = _TicTacToeViT(_CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop.
    # Normally whole dataset is one epoch.
    # However, we have infinite data. So epoch is custom.
    for epoch in range(10):  # number of epochs
        model.train()
        running_loss = 0.0
        for index, (images, labels) in enumerate(
            itertools.islice(train_loader, _SAMPLES_PER_EPOCH)
        ):
            print(f"Epoch: {epoch}, Index: {index}")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {running_loss/_SAMPLES_PER_EPOCH}")

        model.eval()
        # TODO: Can put out-of-sample evaluation code here.


if __name__ == "__main__":
    main()
