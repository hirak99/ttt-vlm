import argparse
import datetime
import json
import logging
import pathlib
import random
import time

import pydantic

from ..llm_service import abstract_llm
from ..llm_service import llm
from ..ttt import ttt_board

_DATA_DIR = pathlib.Path("_data")

_RANDOM_BOARDS_SEED: int | None = None

# This should be whose_move if AI thinks the game ended.
_ENDED = "ended"


class LLMResult(pydantic.BaseModel):
    # Original position.
    board: str
    # When the query was run up to seconds.
    timestamp: int
    # Populated if there was error in response like illegal JSON.
    error: str
    # Model used for the LLM.
    ai_model: str
    # AI guessing whose move.
    ai_whose_move: str
    # AI's move. None if the move is not there because game ended.
    ai_move: list[str] | None
    # How long it took for the AI to get the result.
    duration_ms: int


class _RandomBoardGen:
    def __init__(self, seed: int | None, allow_dups: bool) -> None:
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)

        self._boards_seen: set[ttt_board.BoardState] = set()
        self._allow_dups = allow_dups

    def _random_board_or_none(self) -> ttt_board.BoardState | None:
        board = ttt_board.BoardState.from_array(["."] * 9)
        num_moves = self._rng.randint(1, 6)
        for _ in range(num_moves):
            moves = list(board.allowed_moves())
            if not moves:
                # Game ended before we could have this many moves.
                return None
            board = self._rng.choice(moves)
        return board

    def random_board(self) -> ttt_board.BoardState:
        while True:
            board = self._random_board_or_none()
            if board is not None:
                if self._allow_dups or board not in self._boards_seen:
                    self._boards_seen.add(board)
                    return board
                else:
                    logging.info(f"Already seen {board.as_string()}, trying again.")
            else:
                logging.info("Board is None, trying again.")


class _LlmEvaluator:
    def __init__(self) -> None:
        self._boardgen = _RandomBoardGen(seed=_RANDOM_BOARDS_SEED, allow_dups=True)
        now = datetime.datetime.now()
        self._log_name = _DATA_DIR / now.strftime("%Y%m%d_%H%M%S.jsonl")

    def generate(self, llm_instance: abstract_llm.AbstractLlm):
        board = self._boardgen.random_board()
        prompt_lines = [
            "Given this tic-tac-toe board, please state your next move.",
            f"{json.dumps(board.as_array())}",
            "---",
            "Your output must be a valid json of the following format -",
            "{",
            f'  "whose_move": "X" or "O" or "{_ENDED}",',
            '  "updated_board": [UPDATED_BOARD_AFTER_MOVE]',
            "}",
            'Omit "updated_board" if game has ended.',
        ]
        prompt = "\n".join(prompt_lines)
        start_time = time.time()
        response_json = llm_instance.do_prompt(prompt, max_tokens=1024)
        duration_ms = int((time.time() - start_time) * 1000)

        # Default values if there is any error.
        ai_whose_move: str = ""
        ai_move: list[str] | None = None
        # Check for all kinds of errors.
        error = ""
        try:
            response = json.loads(response_json)
            if not isinstance(response, dict):
                error = "response is not dict"
            else:
                if "whose_move" not in response:
                    error = "whose_move not in response"
                else:
                    whose_move = response["whose_move"]
                    if isinstance(whose_move, str):
                        ai_whose_move = whose_move
                    else:
                        error = "whose_move is not str"
                if "updated_board" in response:
                    updated_board = response["updated_board"]
                    if isinstance(updated_board, list) and all(
                        isinstance(x, str) for x in updated_board
                    ):
                        ai_move = updated_board
                    else:
                        error = "updated_board is not list"

        except json.JSONDecodeError:
            error = "illegal JSON"

        result = LLMResult(
            board=board.as_string(),
            timestamp=int(time.time()),
            error=error,
            ai_model=llm_instance.model_description(),
            ai_whose_move=ai_whose_move,
            ai_move=ai_move,
            duration_ms=duration_ms,
        )

        # Create log dir if it does not exist.
        _DATA_DIR.mkdir(exist_ok=True)

        # Create or append to log.
        with open(self._log_name, "a") as f:
            f.write(result.model_dump_json() + "\n")

        return result


def _generate_data(count: int):
    # llm_instance = llm.OpenAiLlmInstance("gpt-4.1")
    # llm_instance = llm.OpenAiLlmInstance("o3")
    llm_instance = llm.OpenAiLlmInstance("gpt-3.5-turbo")
    tester = _LlmEvaluator()
    for _ in range(count):
        tester.generate(llm_instance)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", help="Number of data points to generate.", type=int, default=10
    )
    args = parser.parse_args()
    _generate_data(args.count)
    logging.info("Done.")
