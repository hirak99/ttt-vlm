import unittest

from . import ttt_evaluator

_TEST_CASES = [
    (
        ("X..|...|..O", "X.O|...|..O"),
        (ttt_evaluator.TttEvaluation.INVALID, "The move does not define valid state."),
    ),
    (
        ("X..|...|..O", "X.X|...|..O"),
        (ttt_evaluator.TttEvaluation.BEST_MOVE, "Best move. X wins in 4 moves."),
    ),
    (
        ("X..|...|...", "X.O|...|..."),
        (ttt_evaluator.TttEvaluation.BLUNDER, "Was draw, now opponent wins. X wins in 5 moves."),
    ),
    (
        ("X.O|...|...", "X.O|.X.|..."),
        (ttt_evaluator.TttEvaluation.BLUNDER, "Was won, but now only a draw. Drawn position."),
    ),
]


class TttEvaluatorTest(unittest.TestCase):

    def test_cases(self):
        evaluator = ttt_evaluator.TttEvaluator()
        for (board0, board1), expected_result in _TEST_CASES:
            acutal = evaluator.evaluate_move(board0, board1)
            self.assertEqual(acutal, expected_result)
