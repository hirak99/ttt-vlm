import unittest

from . import ttt_evaluator

_TEST_CASES = [
    (
        ("X..|...|..O", "X.O|...|..O"),
        (ttt_evaluator.TttEvaluation.ILLEGAL, "The move does not define valid state."),
    ),
    (
        ("X..|...|..O", "X.X|...|..O"),
        (ttt_evaluator.TttEvaluation.BEST_MOVE, "Best move."),
    ),
    (
        ("X..|...|...", "X.O|...|..."),
        (ttt_evaluator.TttEvaluation.GOOD_MOVE, "Good move."),
    ),
    (
        ("X.O|...|...", "X.O|.X.|..."),
        (ttt_evaluator.TttEvaluation.BLUNDER, "Lost the certainty of winning."),
    ),
]


class TttEvaluatorTest(unittest.TestCase):

    def test_cases(self):
        evaluator = ttt_evaluator.TttEvaluator()
        for (board0, board1), expected_result in _TEST_CASES:
            acutal = evaluator.evaluate_move(board0, board1)
            self.assertEqual(acutal, expected_result)
