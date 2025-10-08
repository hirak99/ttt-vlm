import enum
from typing import Sequence

import ttt.ttt_board

from . import ttt_board
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
        # Change sign to match this player.
        board0 = ttt.ttt_board.BoardState.from_array(board_str)
        try:
            board1 = ttt.ttt_board.BoardState.from_array(board_after_move_str)
        except ttt_board.IllegalBoardState:
            # TODO: Add reasons based on some basic checks on why this move is illegal.
            return TttEvaluation.ILLEGAL, "The move does not define valid state."

        reason, message = self._evaluate_move_internal(board0, board1)
        if reason != TttEvaluation.ILLEGAL:
            message = f"{message} {self.solver.solve(board1).text_analysis()}"

        return reason, message

    def _evaluate_move_internal(
        self, board0: ttt.ttt_board.BoardState, board1: ttt.ttt_board.BoardState
    ) -> tuple[TttEvaluation, str]:
        # Evaluate the move.
        # Score based on what is the effective score after move, only if it drops.
        # Score for this player after the move:
        #   <= -900: Blunder
        #   <= -800: Strategic error
        #   <= -700: Miscalculation

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
            # Starting from a winning or drawn position.
                if win_lead1 < 0:
                    # Now lost.
                    if win_lead0 > 0:
                        return (TttEvaluation.BLUNDER, "Hands a won game to the opponent.")
                    else:
                        return (TttEvaluation.BLUNDER, "Was draw, now opponent wins.")
                elif win_lead1 == 0:
                    # Now drawn.
                    if win_lead0 > 0:
                        return (TttEvaluation.BLUNDER, "Was won, but now only a draw.")
                    else:
                        return TttEvaluation.BLUNDER, "Good move. Maintains neutrality."
                elif win_lead1 > 0:
                    # Continues winning.
                    if win_lead1 < win_lead0 - 1:
                        # Gave away a shorter win.
                        return TttEvaluation.GOOD_MOVE, "Good but missed a quicker victory."
                # Note: This may sometimes not be the "best move", depending on forcing.
                # E.g. (1,1), (3,2) leads to "X..|...|.O.". Then (1,3) is less preferable than (3,1).
                # That is due to the quirks of eval function. However, we ignore that and call this best.
                return TttEvaluation.BEST_MOVE, f"Continues to win."
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
            return TttEvaluation.GOOD_MOVE, "Good move, though loss is unavoidable."
        else:
            raise RuntimeError(f"Unexpected value of {win_lead0=}")


def _main():
    solver = ttt_solver.get_instance()
    solver.trace_win(ttt.ttt_board.BoardState.from_array("X..|...|..O"))

    evaluator = TttEvaluator()
    for board, board_after_move in [
        ("X.......O", "X.O.....O"),
        ("X.......O", "X.X.....O"),
        ("X........", "X.O......"),
    ]:
        print(evaluator.evaluate_move(board, board_after_move))


if __name__ == "__main__":
    _main()
