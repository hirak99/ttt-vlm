import argparse
import datetime
import json
import logging
import pathlib
import random
import time

import pydantic

from . import ttt_players
from ..ttt import ttt_board

_DATA_DIR = pathlib.Path("_data")

_RANDOM_BOARDS_SEED: int | None = None


class PlayerResult(pydantic.BaseModel):
    # When the query was run up to seconds.
    timestamp: int
    # Original position.
    board: str
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

    def generate(self, player: ttt_players.AbstractPlayer):
        board = self._boardgen.random_board()
        start_time = time.time()
        response_json = player.play(board)
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

        result = PlayerResult(
            board=board.as_string(),
            timestamp=int(time.time()),
            error=error,
            ai_model=player.model_description(),
            ai_whose_move=ai_whose_move,
            ai_move=ai_move,
            duration_ms=duration_ms,
        )

        return result

    def save(self, result: PlayerResult):
        # Create log dir if it does not exist.
        _DATA_DIR.mkdir(exist_ok=True)

        # Create or append to log.
        with open(self._log_name, "a") as f:
            f.write(result.model_dump_json() + "\n")


def _generate_data(player_name: str, count: int, save_data: bool):
    player = ttt_players.player_factory(player_name)
    tester = _LlmEvaluator()
    for i in range(count):
        logging.info(f"Generating {i+1} of {count}...")
        result = tester.generate(player)
        if save_data:
            tester.save(result)
        else:
            print(result.model_dump_json())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", help="Number of data points to generate.", type=int, default=50
    )
    parser.add_argument(
        "--player",
        help='Either "random", or an openAI model name e.g. "gpt-4.1", "o3", "gpt-3.5-turbo".',
        type=str,
        default="gpt-4.1",
    )
    # Argument to disable saving the data.
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    _generate_data(
        player_name=args.player, count=args.count, save_data=not args.no_save
    )
    logging.info("Done.")
