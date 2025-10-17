import logging
import random

from ..ttt import ttt_board


class RandomBoardGen:
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


def random_board() -> ttt_board.BoardState:
    return RandomBoardGen(seed=None, allow_dups=True).random_board()
