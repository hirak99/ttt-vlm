import importlib
import json

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from src.experiments import vision_performance
from src.llm_service import vision
from src.ttt import board_draw
from src.ttt import ttt_board

from . import board_utils
from .. import misc_utils
from ..llm_service import abstract_llm
from ..llm_service import vision
from ..ttt import board_draw

importlib.reload(vision_performance)
importlib.reload(ttt_board)

_TESTING = True

_GRID_SIZE = (128, 150)
_MARGIN = 4

_PROMPT = """
Please identify this tic-tac-toe position.

You must output a valid json of the form -
[CELL_1, CELL_2, ..., CELL_9]

It should have the 9 cells.
Each CELL_i can be ether "X", "O", or "".
Do not output anything else.
"""


def _ai_recognize(instance: abstract_llm.AbstractLlm, image: Image.Image) -> str:
    return instance.do_prompt(
        _PROMPT, max_tokens=1024, image_b64=vision.to_base64(image)
    )


def _result_to_image(
    board_image: Image.Image, board_actual: ttt_board.BoardState, ai_output: str
):
    image = Image.new("RGB", _GRID_SIZE, (255, 255, 255))
    image_size = (_GRID_SIZE[0] - _MARGIN, _GRID_SIZE[0] - _MARGIN)
    image.paste(board_image.resize(image_size), (_MARGIN, _MARGIN))
    draw = ImageDraw.Draw(image)
    correct = False
    try:
        ai_board_json = json.loads(ai_output)
        ai_board = ttt_board.BoardState.from_array(ai_board_json)
        ai_text = "Detected: " + ai_board.as_string()
        correct = board_actual == ai_board
    except json.JSONDecodeError:
        ai_text = f"Bad JSON: {ai_output!r}"
    except ttt_board.IllegalBoardState as e:
        ai_text = f"Illegal: {e}"
    draw.circle((10, 10), radius=5, fill=(0, 196, 0) if correct else (255, 0, 0))
    font = ImageFont.truetype(
        misc_utils.pil_font(), size=11
    )
    draw.text(
        xy=(_GRID_SIZE[0] // 2, _GRID_SIZE[0] + _MARGIN),
        text=ai_text,
        fill=(0, 0, 0),
        font=font,
        anchor="mt",
    )
    return image


def test_random_board() -> Image.Image:
    board = board_utils.random_board()
    render_params = board_draw.RenderParams.random()
    image = board_draw.to_image(board.as_array(), render_params)
    if _TESTING:
        recognized = '["O", "", "O", "", "O", "X", "X", "", "X"]'
    else:
        instance = vision.OpenAiVision()
        recognized = _ai_recognize(instance, image)
    return _result_to_image(image, board, recognized)
