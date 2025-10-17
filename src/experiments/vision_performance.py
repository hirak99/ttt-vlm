from . import board_utils
from ..llm_service import vision
from ..ttt import board_draw

_PROMPT = """
Please identify this tic-tac-toe position.

You must output a valid json of the form -
[CELL_1, CELL_2, ..., CELL_9]

It should have the 9 cells.
Each CELL_i can be ether "X", "O", or "".
Do not output anything else.
"""


def _evaluate_wip():
    board = board_utils.random_board()
    print(board.as_array())
    render_params = board_draw.RenderParams.random()
    image = board_draw.to_image(board.as_array(), render_params)
    instance = vision.OpenAiVision()
    return instance.do_prompt(
        _PROMPT, max_tokens=1024, image_b64=vision.to_base64(image)
    )


if __name__ == "__main__":
    _evaluate_wip()
