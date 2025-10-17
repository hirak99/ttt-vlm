from . import board_utils
from ..llm_service import abstract_llm
from ..llm_service import vision
from ..ttt import board_draw
from ..ttt import ttt_board

_PROMPT = """
Please identify this tic-tac-toe position.

You must output a valid json of the form -
[CELL_1, CELL_2, ..., CELL_9]

It should have the 9 cells.
Each CELL_i can be ether "X", "O", or "".
Do not output anything else.
"""


def _ai_recognize(
    instance: abstract_llm.AbstractLlm,
    board: ttt_board.BoardState,
    render_params: board_draw.RenderParams,
) -> str:
    print(board.as_array())
    image = board_draw.to_image(board.as_array(), render_params)
    return instance.do_prompt(
        _PROMPT, max_tokens=1024, image_b64=vision.to_base64(image)
    )


def main():
    board = board_utils.random_board()
    render_params = board_draw.RenderParams.random()
    instance = vision.OpenAiVision()
    _ai_recognize(instance, board, render_params)


if __name__ == "__main__":
    main()
