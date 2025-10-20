import logging
import pathlib

import safetensors.torch
from torch import nn

# This is the number of cells.
# True constant - this will not change.
NUM_CLASSES = 9

# Saved after all epochs are done.
FINAL_MODEL_FILE = pathlib.Path("_data") / "custom_model.safetensor"


class TicTacToeVision(nn.Module):
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, NUM_CLASSES, 3)
        return x

    def save_savetensors(self, fname: pathlib.Path = FINAL_MODEL_FILE):
        safetensors.torch.save_file(self.state_dict(), fname)
        logging.info(f"Saved model to {fname}")

    def load_safetensors(self, fname: pathlib.Path = FINAL_MODEL_FILE):
        self.load_state_dict(safetensors.torch.load_file(fname))
        logging.info(f"Loaded model from {fname}")
