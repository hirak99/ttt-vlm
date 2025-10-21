from . import base_model
from . import models_internal
from typing import Type

_REGISTRY: dict[str, Type[base_model.BaseModel]] = {
    "cnnv1": models_internal.CnnV1,
    "cnnv2": models_internal.CnnV2,
}


def get_model(model_name: str) -> base_model.BaseModel | None:
    if model_name not in _REGISTRY:
        return None
    model = _REGISTRY[model_name]()
    model.file_suffix = model_name
    return model
