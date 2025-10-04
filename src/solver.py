import dataclasses
import functools
import logging
from typing import Iterator


def _sign(x: float) -> int:
    return 1 if x >= 0 else -1 if x < 0 else 0


@dataclasses.dataclass(frozen=True)
class BoardState:
    xs: int
    os: int

    x_to_move: bool

    @functools.cached_property
    def emptys(self) -> int:
        return 0b111111111 & ~self.xs & ~self.os

    def as_string(self) -> str:
        # Returns XX.|.OX|OOO
        board: list[str] = []
        for n in range(9):
            if self.xs & (1 << n):
                board.append("X")
            elif self.os & (1 << n):
                board.append("O")
            else:
                board.append(".")
            if n < 8 and (n + 1) % 3 == 0:
                board.append("|")
        return "".join(board)

    @classmethod
    def from_string(cls, board_str: str) -> "BoardState":
        board_str = board_str.replace("|", "")
        if len(board_str) != 9:
            raise ValueError(f"Must be 9 characters long: {board_str}")
        if not all(c in "XO." for c in board_str):
            raise ValueError(f"Must be X, O or .: {board_str}")
        xs = 0
        os = 0
        x_count = 0
        o_count = 0
        for n, c in enumerate(board_str):
            if c == "X":
                xs |= 1 << n
                x_count += 1
            elif c == "O":
                os |= 1 << n
                o_count += 1
        if x_count != o_count and x_count != o_count + 1:
            raise ValueError(f"X's must be same or more than O's: {board_str}")
        return BoardState(xs, os, x_count == o_count)


@dataclasses.dataclass(frozen=True)
class StateScore:
    score: int
    best_plays: list[BoardState]


class AllStates:
    def __init__(self):
        self._all_scores: dict[BoardState, StateScore] = {}

    def allowed_moves(self, state: BoardState) -> Iterator[BoardState]:
        for n in range(9):
            if state.emptys & (1 << n):
                new_xs = state.xs | (1 << n) if state.x_to_move else state.xs
                new_os = state.os | (1 << n) if not state.x_to_move else state.os
                yield BoardState(new_xs, new_os, not state.x_to_move)

    def terminal_score(self, state: BoardState) -> int | None:
        # +10 if game ended and X has won.
        # -10 if game ended and O has won.
        # 0 if game ended and no one has won.
        # None if game has not ended.
        def check_ones(against: int, ones: int) -> bool:
            return ones & against == ones

        for score, occ in [(10, state.xs), (-10, state.os)]:
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

    # +10 if X has won.
    # -10 if O has won.
    # +10 - k if X wins in k moves.
    # -10 + k if O wins in k moves.
    # 0 if no one has perfect play.
    def solve_for_score(self, state: BoardState) -> int:
        if state in self._all_scores:
            return self._all_scores[state].score
        solved_score = self.terminal_score(state)
        best_plays: list[BoardState] = []
        if solved_score is None:
            best_score = None
            for next_state in self.allowed_moves(state):
                score = self.solve_for_score(next_state)
                score = (
                    abs(score) + len(self._all_scores[next_state].best_plays) / 10.0
                ) * _sign(score)
                if score != 0:
                    # Reduce in asbsolute value by one.
                    score -= 1 if score > 0 else -1
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
        self._all_scores[state] = StateScore(solved_score, best_plays)
        logging.info(
            f"Explored positions: {len(self._all_scores)}, last state: {state.as_string()} score {solved_score}"
        )
        return self._all_scores[state].score

    def trace_win(self, state: BoardState) -> None:
        self.solve_for_score(state)
        print(f"Score: {self._all_scores[state].score}")
        if self._all_scores[state].score == 0:
            print("Position is a draw")
        print(state.as_string())
        while self._all_scores[state].best_plays:
            next_state = self._all_scores[state].best_plays[0]
            state = next_state
            print(state.as_string())


def __main__():
    all_states = AllStates()
    # all_states.trace_win(BoardState.from_string("...|...|..."))
    # all_states.trace_win(BoardState.from_string("X.X|O..|O.X"))
    # all_states.trace_win(BoardState.from_string("XO.|.O.|OXX"))
    all_states.trace_win(BoardState.from_string("...|O..|.XX"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    __main__()
