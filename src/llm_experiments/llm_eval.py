import json
import logging
import random

from ..llm_service import llm
from ..ttt import ttt_board


class RandomBoardGen:
    def __init__(self) -> None:
        self._rng = random.Random()
        self._rng.seed(42)

        self._boards_seen: set[ttt_board.BoardState] = set()

    def _random_board_or_none(self) -> ttt_board.BoardState | None:
        board = ttt_board.BoardState.from_array(["."] * 9)
        num_moves = self._rng.randint(0, 6)
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
                if board not in self._boards_seen:
                    self._boards_seen.add(board)
                    return board
                else:
                    logging.info(f"Already seen {board.as_string()}, trying again.")
            else:
                logging.info("Board is None, trying again.")


def main():
    boardgen = RandomBoardGen()

    # for _ in range(100):
    #     print(boardgen.random_board().as_string())

    instance = llm.OpenAiLlmInstance("gpt-4.1")
    prompt_lines = [
        "Given this tic-tac-toe board, please state your next move.",
        f"{json.dumps(boardgen.random_board().as_array())}",
        "---",
        "Your output must be a valid json of the following format -",
        "{",
        '  "whose_move": "X" or "O" or "ended",',
        '  "updated_board": [UPDATED_BOARD_AFTER_MOVE]',
        "}",
        'Omit "updated_board" if game has ended.',
    ]
    prompt = "\n".join(prompt_lines)
    print("Prompt:\n", prompt)
    print("Response:\n", instance.do_prompt(prompt, max_tokens=1024))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    logging.info("Done.")
