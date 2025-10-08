import dataclasses
import functools
from typing import Iterator, Sequence


class IllegalBoardState(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class BoardState:
    """State of a baord.

    Note: Prefer instantiating through the factory methods over direct instantiation.
    """

    xs: int
    os: int

    x_to_move: bool

    @functools.cached_property
    def move_count(self) -> int:
        return bin(self.xs | self.os).count("1")

    @functools.cached_property
    def emptys(self) -> int:
        return 0b111111111 & ~self.xs & ~self.os

    def allowed_moves(self) -> Iterator["BoardState"]:
        for n in range(9):
            if self.emptys & (1 << n):
                new_xs = self.xs | (1 << n) if self.x_to_move else self.xs
                new_os = self.os | (1 << n) if not self.x_to_move else self.os
                yield BoardState(new_xs, new_os, not self.x_to_move)

    def as_array(self) -> list[str]:
        # Returns ['X', 'X', '.', '.', 'O', 'X', 'O', 'O', 'O'].
        board: list[str] = []
        for n in range(9):
            if self.xs & (1 << n):
                board.append("X")
            elif self.os & (1 << n):
                board.append("O")
            else:
                board.append(".")
        return board

    def as_string(self) -> str:
        # Returns XX.|.OX|OOO
        board: list[str] = self.as_array()
        board.insert(6, "|")
        board.insert(3, "|")
        return "".join(board)

    @classmethod
    def from_array(cls, board: Sequence[str]) -> "BoardState":
        """Constructs a partially filled TTT board.

        Args:
            board: List or str of length 9 with X, O or . Character '|' can be
                used for separation and will be ignored. For example "XX.|.OX|OOO".
        """
        if isinstance(board, str):
            board = board.replace("|", "")
            if len(board) != 9:
                raise IllegalBoardState(f"Must be 9 characters long: {board}")
            if not all(c in "XO." for c in board):
                raise IllegalBoardState(f"Must be X, O or .: {board}")

        xs = 0
        os = 0
        x_count = 0
        o_count = 0
        for n, c in enumerate(board):
            if c == "X":
                xs |= 1 << n
                x_count += 1
            elif c == "O":
                os |= 1 << n
                o_count += 1
            elif c != ".":
                raise IllegalBoardState(f"Must be X, O or .: {board}")
        if x_count != o_count and x_count != o_count + 1:
            raise IllegalBoardState(f"X's must be same or more than O's: {board}")
        return cls(xs, os, x_count == o_count)
