"""Image recognizers.

Factory for functions which take a tic-tac-toe image, and returns a string with
all the identified cells.
"""

import functools

from PIL import Image

from ..llm_service import abstract_llm
from ..llm_service import vision

from typing import Callable

_PROMPT = """
Please identify this tic-tac-toe position.

You must output a valid json of 9 characters -
[CELL_1, CELL_2, ..., CELL_9]

Each CELL_i can be ether "X", "O", or "", read in standard reading order.

Output just the JSON. Do not output anything else.
"""

RecognizeFnT = Callable[[Image.Image], str]


def _vlm_recognize_fn(instance: abstract_llm.AbstractLlm) -> RecognizeFnT:

    def recognize(image: Image.Image, instance: abstract_llm.AbstractLlm) -> str:
        return instance.do_prompt(
            _PROMPT, max_tokens=1024, image_b64=vision.to_base64(image)
        )

    return functools.partial(recognize, instance=instance)


def get_recognizer(recognizer_type: str) -> tuple[str, RecognizeFnT]:

    # VLM recognizers.
    if recognizer_type.startswith("gpt") or recognizer_type.startswith("o3"):
        instance = vision.OpenAiVision(recognizer_type)
    elif recognizer_type.startswith("blaifa"):
        instance = vision.OllamaVision(recognizer_type)
    else:
        raise ValueError(f"Unknown model: {recognizer_type}")

    return instance.model_description(), _vlm_recognize_fn(instance)
