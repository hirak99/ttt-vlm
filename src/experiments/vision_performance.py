import datetime
import logging
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from . import board_utils
from .. import misc_utils
from ..llm_service import abstract_llm
from ..llm_service import llm_utils
from ..llm_service import vision
from ..ttt import board_draw
from ..ttt import ttt_board

_RESULT_DIR = "_data/vision"

_GRID_SIZE = (128, 158)
_MARGIN = 4

_PROMPT = """
Please identify this tic-tac-toe position.

You must output a valid json of 9 characters -
[CELL_1, CELL_2, ..., CELL_9]

Each CELL_i can be ether "X", "O", or "", read in standard reading order.

Output just the JSON. Do not output anything else.
"""

# If true, does not actually query the LLM.
_SKIP_LLM_FOR_TESTS = False


def _ai_recognize(instance: abstract_llm.AbstractLlm, image: Image.Image) -> str:
    return instance.do_prompt(
        _PROMPT, max_tokens=1024, image_b64=vision.to_base64(image)
    )


def _result_to_image(
    board_image: Image.Image, board_actual: ttt_board.BoardState, ai_output: str
) -> tuple[bool, Image.Image]:
    image = Image.new("RGB", _GRID_SIZE, (255, 255, 255))
    image_size = (_GRID_SIZE[0] - _MARGIN, _GRID_SIZE[0] - _MARGIN)
    image.paste(board_image.resize(image_size), (_MARGIN, _MARGIN))
    draw = ImageDraw.Draw(image)
    correct = False

    text_color: tuple[int, int, int]
    status: str
    # TODO: These should be passed as transformers to do_prompt_and_parse().
    without_think = llm_utils.remove_thinking(ai_output)
    ai_board_json = llm_utils.parse_as_json(without_think)
    if ai_board_json is None:
        status, ai_text = "JSON Error:", str(without_think)
        text_color = (64, 64, 128)  # JSON Error.
    else:
        try:
            ai_board = ttt_board.BoardState.from_array(ai_board_json)
            correct = board_actual == ai_board
            if correct:
                status, ai_text = "Detected (correct):", ai_board.as_string()
                text_color = (0, 128, 0)  # Correct.
            else:
                status, ai_text = "Detected (incorrect):", ai_board.as_string()
                text_color = (255, 0, 0)  # Incorrect.
        except ttt_board.IllegalBoardState as e:
            status, ai_text = "Illegal:", str(e)
            text_color = (128, 64, 0)  # Illegal.

    logging.info(f"Result: {ai_text}")
    correctness_color = (0, 196, 0) if correct else (255, 0, 0)
    draw.circle((10, 10), radius=5, fill=correctness_color)
    font = ImageFont.truetype(misc_utils.pil_font(), size=11)

    # We don't want multiline text.
    ai_text = ai_text.replace("\n", "<cr>")

    draw.text(
        xy=(0, _GRID_SIZE[0] + _MARGIN),
        text=status + "\n" + ai_text,
        fill=text_color,
        font=font,
    )
    return correct, image


def _random_board_eval(
    llm_instance: abstract_llm.AbstractLlm,
) -> tuple[bool, Image.Image]:
    board = board_utils.random_board()
    logging.info(f"Random board: {board.as_array()}")
    render_params = board_draw.RenderParams.random()
    image = board_draw.to_image(board.as_array(), render_params)
    if _SKIP_LLM_FOR_TESTS:
        recognized = '["O", "", "O", "", "O", "X", "X", "", "X"]'
    else:
        recognized = _ai_recognize(llm_instance, image)
    logging.info(f"AI Output: {recognized!r}")
    return _result_to_image(image, board, recognized)


def random_eval_grid(
    header: str, llm_instance: abstract_llm.AbstractLlm, rows: int, cols: int
) -> Image.Image:
    header_height = 40

    # Call the test_random_board to get the size of a single board
    correct, sample_board = _random_board_eval(llm_instance)
    board_width, board_height = sample_board.size

    # Create a new blank image large enough to hold all the boards
    grid_width = cols * board_width
    grid_height = rows * board_height + header_height

    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Loop through each cell in the grid
    success_counter = 0
    for row in range(rows):
        for col in range(cols):
            logging.info(f"Test {row * cols + col + 1} of {rows * cols} ...")
            if row == 0 and col == 0:
                board = sample_board
            else:
                correct, board = _random_board_eval(llm_instance)

            if correct:
                success_counter += 1

            # Calculate where to paste this board in the grid
            x_offset = col * board_width
            y_offset = row * board_height + header_height

            # Paste the board into the grid
            grid_image.paste(board, (x_offset, y_offset))

    # Write header.
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype(misc_utils.pil_font(), size=16)
    draw.text(
        xy=(_MARGIN, 0),
        text=f"{header}\nCorrect: {success_counter}/{rows*cols}",
        fill=(0, 0, 0),
        font=font,
    )

    return grid_image


def main():
    logging.basicConfig(level=logging.INFO)

    instance = vision.OllamaVision("blaifa/InternVL3_5:8b")
    # instance = vision.OpenAiVision("gpt-4.1")

    image = random_eval_grid("Model: " + instance.model_description(), instance, 10, 10)
    # Date time for file suffix.
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfname = os.path.join(_RESULT_DIR, f"result_grid_{suffix}.png")
    os.makedirs(_RESULT_DIR, exist_ok=True)
    image.save(outfname)
    logging.info(f"Saved to {outfname}")


if __name__ == "__main__":
    main()
