from . import base_model
from . import models_internal

from typing import Type

# For convenience, this serves as the default model used in training,
# evaluation, etc.
DEFAULT_MODEL_NAME: str = "cnnv3"


# Can add one off models here.
_REGISTRY: dict[str, Type[base_model.BaseModel]] = {}

# Add models defined in other modules to the registry.
_REGISTRY.update(models_internal.for_registry())


class ModelNotFoundError(Exception):
    pass


def get_model(model_name: str) -> base_model.BaseModel:
    if model_name not in _REGISTRY:
        raise ModelNotFoundError(f"Unknown model: {model_name}")
    model = _REGISTRY[model_name]()
    model.file_suffix = model_name
    return model


def default_model() -> base_model.BaseModel | None:
    return get_model(DEFAULT_MODEL_NAME)
