"""Image recognizers.

Factory for functions which take a tic-tac-toe image, and returns a string with
all the identified cells.
"""

import functools
import json

from PIL import Image

from ..llm_service import abstract_llm
from ..llm_service import vision
from .model import registry

from typing import Callable

_PROMPT = """
Please identify this tic-tac-toe position.

You must output a valid json of 9 characters -
[CELL_1, CELL_2, ..., CELL_9]

Each CELL_i can be ether "X", "O", or "", read in standard reading order.

Output just the JSON. Do not output anything else.
"""

# NOTE: The return value must be a valid json. It will be parsed and read as list[str].
RecognizeFnT = Callable[[Image.Image], str]


def _vlm_recognize_fn(instance: abstract_llm.AbstractLlm) -> RecognizeFnT:

    def recognize(image: Image.Image, instance: abstract_llm.AbstractLlm) -> str:
        return instance.do_prompt(
            _PROMPT, max_tokens=1024, image_b64=vision.to_base64(image)
        )

    return functools.partial(recognize, instance=instance)


def get_recognizer(recognizer_type: str) -> tuple[str, RecognizeFnT]:
    """Factory to return a recognizer function.

    Args:
      recognizer_type: A string indicating what to instantiate.
        Examples include -
        - "custom_model"
        - "blaifa/InternVL3_5:8b"
        - "gpt-4.1"
        - "o3"
    """
    # First check if we have a custom model by this name.
    model = registry.get_model(recognizer_type)
    if model is not None:
        model.load_safetensors()
        return "Custom Model", lambda image: json.dumps(model.recognize(image))

    # Then check for VLM recognizers.
    if recognizer_type.startswith("gpt") or recognizer_type.startswith("o3"):
        instance = vision.OpenAiVision(recognizer_type)
    elif recognizer_type.startswith("blaifa"):
        instance = vision.OllamaVision(recognizer_type)
    else:
        raise ValueError(f"Unknown model: {recognizer_type}")

    return instance.model_description(), _vlm_recognize_fn(instance)
