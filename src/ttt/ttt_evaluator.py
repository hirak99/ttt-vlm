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
        self.solver = ttt_solver.get_instance()

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

        for best_move in self.solver.best_moves(board0):
            if best_move == board1:
                return TttEvaluation.BEST_MOVE, "Best move."

        sign = 1 if board0.x_to_move else -1

        sol0 = self.solver.solve(board0)
        sol1 = self.solver.solve(board1)
        # +/-10 indicates me/opponent already won.
        # A lower positive number indicates win in next (10 - abs(n)) moves.
        win_lead0 = sign * sol0.win_lead
        win_lead1 = sign * sol1.win_lead

        if win_lead0 >= 0:
            if win_lead0 > 0:
                # Was winning.
                if win_lead1 < 0:
                    # Now lost.
                    return (TttEvaluation.BLUNDER, "Hands a won game to the opponent.")
                elif win_lead1 == 0:
                    # Now drawn.
                    return TttEvaluation.BLUNDER, "Lost the certainty of winning."
                elif win_lead1 < win_lead0 - 1:
                    # Gave away a shorter win.
                    return TttEvaluation.GOOD_MOVE, "Good but missed a quicker victory."
            return TttEvaluation.GOOD_MOVE, "Good move."
        elif win_lead0 < 0:
            # Evaluation starting from a losing position.
            if win_lead1 < win_lead0 - 1:
                # Hastens the loss.
                if win_lead1 <= -9:
                    # Loses immediately.
                    return (
                        TttEvaluation.BLUNDER,
                        "Blunder - hastens to immediate defeat.",
                    )
                if win_lead1 <= -8:
                    return (
                        TttEvaluation.STRATEGIC_ERROR,
                        "Strategic error - hastens defeat.",
                    )
                if win_lead1 <= -7:
                    return TttEvaluation.MISCALCULATION, "Miscalculation"
            return TttEvaluation.GOOD_MOVE, "Good move"
        else:
            raise RuntimeError(f"Unexpected value of {win_lead0=}")


def _main():
    solver = ttt_solver.get_instance()
    solver.trace_win(ttt_solver.BoardState.from_array("X..|...|..O"))

    evaluator = TttEvaluator()
    for board, board_after_move in [
        ("X.......O", "X.O.....O"),
        ("X.......O", "X.X.....O"),
        ("X........", "X.O......"),
    ]:
        print(evaluator.evaluate_move(board, board_after_move))


if __name__ == "__main__":
    _main()
