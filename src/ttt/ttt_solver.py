import dataclasses
import functools
import logging
from typing import Iterator, Sequence

# Scoring system.
# Scores are always positive for X win. Negate to get scores for O win.
# If the current position is a win.
_SCORE_WIN = 1000
# If it takes n moves to win, n times this will be removed from _SCORE_WIN.
_SCORE_DELAY = -100
# If there are multiple ways to win, this will be added for each ways.
# This incentivizes blocking instead of giving up when there are multiple paths for opponent to win.
_SCORE_NUM_WAYS = 1


class IllegalBoardState(Exception):
    pass


def _sign(x: float) -> int:
    return 1 if x >= 0 else -1 if x < 0 else 0


@dataclasses.dataclass(frozen=True)
class BoardState:
    """State of a baord.

    Note: Prefer instantiating through the factory methods over direct instantiation.
    """

    xs: int
    os: int

    x_to_move: bool

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


@dataclasses.dataclass(frozen=True)
class BoardIntel:
    # Score without counting number of ways to win.
    _base_score: int
    # All possible plays to achieve the score.
    best_plays: list[BoardState]

    @property
    def score(self) -> int:
        return (
            abs(self._base_score)
            # Lower cap at 0 for end state with no plays.
            + max(0, len(self.best_plays) - 1) * _SCORE_NUM_WAYS
        ) * _sign(self._base_score)

    @functools.cached_property
    def win_lead(self) -> int:
        # 0 indicates no win in sight.
        # +10 or -10 indicates X or O has won.
        # Any other number indicates X or O will win in (10 - abs(n)) moves.
        assert _SCORE_DELAY < 0
        # Base score is approximately 100 * what we want. The 100 is -_SCORE_DELAY.
        # Round it to nearest number of moves.
        return (self._base_score + (-_SCORE_DELAY // 2)) // (-_SCORE_DELAY)


class _TttSolver:
    def __init__(self):
        self._all_scores: dict[BoardState, BoardIntel] = {}

    def terminal_score(self, state: BoardState) -> int | None:
        # _SCORE_WIN if game ended and X won.
        # -_SCORE_WIN if game ended and O won.
        # 0 if game ended and no one has won.
        # None if game has not ended.
        def check_ones(against: int, ones: int) -> bool:
            return ones & against == ones

        for score, occ in [(_SCORE_WIN, state.xs), (-_SCORE_WIN, state.os)]:
            # Horizontal.
            if (
                check_ones(occ, 0b111)
                or check_ones(occ, 0b111000)
                or check_ones(occ, 0b111000000)
            ):
                return score
            # Vertical.
            if (
                check_ones(occ, 0b1001001)
                or check_ones(occ, 0b10010010)
                or check_ones(occ, 0b100100100)
            ):
                return score
            # Diagonal.
            if check_ones(occ, 0b100010001) or check_ones(occ, 0b001010100):
                return score
        if state.emptys == 0:
            return 0
        return None

    def solve_for_score(self, state: BoardState) -> int:
        if state in self._all_scores:
            return self._all_scores[state].score
        solved_score = self.terminal_score(state)
        best_plays: list[BoardState] = []
        if solved_score is None:
            best_score = None
            for next_state in state.allowed_moves():
                score = self.solve_for_score(next_state)
                score_sign = _sign(score)
                score = max(0, abs(score) + _SCORE_DELAY) * score_sign
                # If x_to_play then maximize, if not then minimize.
                if (
                    best_score is None
                    or (state.x_to_move and score > best_score)
                    or (not state.x_to_move and score < best_score)
                ):
                    best_score = score
                    best_plays = []
                if score == best_score:
                    best_plays.append(next_state)
            solved_score = best_score
        assert solved_score is not None
        self._all_scores[state] = BoardIntel(solved_score, best_plays)
        logging.info(
            f"Explored positions: {len(self._all_scores)}, last state: {state.as_string()} score {self._all_scores[state].score}"
        )
        return self._all_scores[state].score

    def solve(self, state: BoardState) -> BoardIntel:
        self.solve_for_score(state)
        return self._all_scores[state]

    def best_moves(self, state: BoardState) -> list[BoardState]:
        self.solve_for_score(state)
        return self._all_scores[state].best_plays

    def trace_win(self, state: BoardState) -> None:
        self.solve_for_score(state)
        print(f"Starting trace for: {state.as_string()}")
        while True:
            print(
                f"{state.as_string()} score={self._all_scores[state].score}, continue={len(self._all_scores[state].best_plays)}"
            )
            best_plays = self._all_scores[state].best_plays
            if not best_plays:
                break
            state = best_plays[0]


def get_instance() -> _TttSolver:
    return _TttSolver()


def __main__():
    solver = get_instance()
    solver.trace_win(BoardState.from_array("...|...|..."))
    # all_states.trace_win(BoardState.from_array(".X.|.O.|..."))
    # all_states.trace_win(BoardState.from_array("XXO|OOX|XO."))
    solver.trace_win(BoardState.from_array("X..|...|..O"))
    # all_states.trace_win(BoardState.from_array("...|O..|.XX")) # -501
    # all_states.trace_win(BoardState.from_array("X.X|OO.|OXX"))  # -900
    # all_states.trace_win(BoardState.from_array("XX.|OO.|OXX"))  # -901


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    __main__()
