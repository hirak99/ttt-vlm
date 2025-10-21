import abc
import logging
import pathlib

from PIL import Image
import safetensors.torch
import torch
from torch import nn

# This is the number of cells.
# True constant - this will not change.
NUM_CLASSES = 9

CHAR_TO_CLASSID = {
    "X": 0,
    "O": 1,
    ".": 2,
}
_CLASSID_TO_CHAR = {v: k for k, v in CHAR_TO_CLASSID.items()}


class BaseModel(nn.Module, abc.ABC):

    @classmethod
    @abc.abstractmethod
    def image_to_input(cls, image: Image.Image) -> torch.Tensor:
        pass

    def __init__(self):
        self.file_suffix: str = "unnamed"
        super().__init__()

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

    def save_savetensor(self):
        fname = pathlib.Path("_data") / f"model_{self.file_suffix}.safetensor"
        safetensors.torch.save_file(self.state_dict(), fname)
        logging.info(f"Saved model to {fname}")

    def load_safetensor(self):
        fname = pathlib.Path("_data") / f"model_{self.file_suffix}.safetensor"
        self.load_state_dict(safetensors.torch.load_file(fname))
        logging.info(f"Loaded model from {fname}")
