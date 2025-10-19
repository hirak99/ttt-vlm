from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from transformers import ViTConfig
from transformers import ViTModel

_IMAGE_SIZE = 32


class TicTacToeDataset(IterableDataset[tuple[Image.Image, torch.Tensor]]):
    # Custom dataset for loading Tic-Tac-Toe images and labels
    def __init__(self, transform: transforms.Compose | None = None):
        self._transform = transform

    def __iter__(self):
        while True:
            # yield Image.new("RGB", (32, 32)), torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
            image, label = Image.new("RGB", (32, 32)), torch.tensor(["."])
            image = self._transform(image) if self._transform else image
            yield image, label


# Configuration to instantiate ViTModel.
_CONFIG = ViTConfig(
    image_size=_IMAGE_SIZE,
    patch_size=8,
    num_classes=9,  # 9 output classes for Tic-Tac-Toe grid
    dim=256,  # You can tune this
    depth=6,  # Number of transformer layers
    heads=8,  # Attention heads
    mlp_dim=512,  # Feed-forward layers
)


# Vision Transformer Model
class TicTacToeViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super(TicTacToeViT, self).__init__()
        self.vit = ViTModel(config)
        self.fc = nn.Linear(
            config.dim, 9
        )  # 9 output classes for each cell in Tic-Tac-Toe

    def forward(self, x):
        x = self.vit(x).last_hidden_state
        x = x.mean(dim=1)  # Global average pooling across patches
        x = self.fc(x)  # Final classification
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = TicTacToeViT(_CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):  # number of epochs
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
