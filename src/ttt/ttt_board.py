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

    # Bit field to denote where X's are.
    xs: int
    # Bit field to denote where O's are.
    os: int

    @functools.cached_property
    def digest(self) -> int:
        """An integer representation of this position."""
        result = 0
        # Get base-3 representation.
        # Assume X's and O's bitfields as base 3.
        # Then add X_3 + 2 * O_3.
        for pos in range(9):
            if self.xs & (1 << pos):
                result += 3**pos
            if self.os & (1 << pos):
                result += 2 * 3**pos
        return result

    @classmethod
    def from_digest(cls, digest: int) -> "BoardState":
        """Reconstruct the board from digest."""
        xs = 0
        os = 0
        for pos in range(9):
            bits = digest % 3
            if bits == 1:
                xs |= 1 << pos
            elif bits == 2:
                os |= 1 << pos
            digest //= 3
        return cls(xs, os)

    @functools.cached_property
    def x_to_move(self) -> bool:
        return self.move_count % 2 == 0

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
                yield BoardState(new_xs, new_os)

    @functools.cached_property
    def ended(self) -> bool:
        try:
            next(self.allowed_moves())
        except StopIteration:
            return True
        return False

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
            illegal_chars = [c for c in board if c not in "XO."]
            if illegal_chars:
                raise IllegalBoardState(
                    f"Must be X, O or ., but found {illegal_chars} in {board}"
                )

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
            elif c not in [".", ""]:
                raise IllegalBoardState(f"Must be X, O or .: {board}")
        if x_count != o_count and x_count != o_count + 1:
            raise IllegalBoardState(f"X's must be same or more than O's: {board}")
        return cls(xs, os)
