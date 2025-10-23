import dataclasses
from torch import nn


@dataclasses.dataclass
class ModelSize:
    parameters: int
    size_in_bytes: int

    @property
    def size_in_mb(self) -> float:
        return self.size_in_bytes / 1024 / 1024


def get_model_size(model: nn.Module) -> ModelSize:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = params * 4
    return ModelSize(parameters=params, size_in_bytes=size_in_bytes)
