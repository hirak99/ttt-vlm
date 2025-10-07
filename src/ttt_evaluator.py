import enum
from typing import Sequence

from . import ttt_solver


class TttEvaluation(enum.Enum):
    NOT_EVALUATED = enum.auto()

    ILLEGAL = enum.auto()
    BLUNDER = enum.auto()
    STRATEGIC_ERROR = enum.auto()
    MISCALCULATION = enum.auto()
    GOOD_MOVE = enum.auto()
    BEST_MOVE = enum.auto()


class TttEvaluator:
    def __init__(self):
        self.solver = ttt_solver.TttSolver()

    def evaluate_move(
        self, board_str: Sequence[str], board_after_move_str: Sequence[str]
    ) -> tuple[TttEvaluation, str]:
        # Evaluate the move.
        # Score based on what is the effective score after move, only if it drops.
        # Score for this player after the move:
        #   <= -900: Blunder
        #   <= -800: Strategic error
        #   <= -700: Miscalculation

        # Change sign to match this player.
        board0 = ttt_solver.BoardState.from_array(board_str)
        try:
            board1 = ttt_solver.BoardState.from_array(board_after_move_str)
        except ttt_solver.IllegalBoardState:
            # TODO: Add reasons based on some basic checks on why this move is illegal.
            return TttEvaluation.ILLEGAL, "The move does not define valid state."

        # Check if it is illegal.
        for legal_move in board0.allowed_moves():
            if legal_move == board1:
                break
        else:
            # TODO: Add reasons based on some basic checks on why this move is illegal.
            return TttEvaluation.ILLEGAL, "The move is illegal."

        sign = 1 if board0.x_to_move else -1
        score0 = sign * self.solver.solve_for_score(board0)
        score1 = sign * self.solver.solve_for_score(board1)


def _main():
    solver = ttt_solver.TttSolver()
    solver.trace_win(ttt_solver.BoardState.from_string("X..|...|..O"))

    evaluator = TttEvaluator()
    for board, board_after_move in [
        ("X.......O", "X.O.....O"),
        ("X.......O", "X.X.....O"),
        ("X........", "X.O......"),
    ]:
        print(evaluator.evaluate_move(board, board_after_move))


if __name__ == "__main__":
    _main()
