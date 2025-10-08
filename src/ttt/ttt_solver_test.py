import dataclasses
import unittest

from . import ttt_board
from . import ttt_solver


@dataclasses.dataclass
class _Expect:
    board_str: str
    score: int | None = None
    win_lead: int | None = None
    best_moves: set[str] = dataclasses.field(default_factory=set)


_BOARDS = [
    "...|...|...",
    "X..|...|...",
    "X..|...|..O",
    "X.O|...|..O",
    "X.X|...|..O",
    "X.O|.X.|...",
    "X.O|X.O|.X.",
]


_EXPECTED_RESULTS = [
    _Expect(
        board_str="...|...|...",
        score=8,
        win_lead=0,
        best_moves={
            "..X|...|...",
            "...|.X.|...",
            "...|..X|...",
            "X..|...|...",
            "...|...|X..",
            ".X.|...|...",
            "...|...|.X.",
            "...|X..|...",
            "...|...|..X",
        },
    ),
    _Expect(board_str="X..|...|...", score=0, win_lead=0, best_moves={"X..|.O.|..."}),
    _Expect(
        board_str="X..|...|..O",
        score=502,
        win_lead=5,
        best_moves={"X.X|...|..O", "X..|...|X.O"},
    ),
    _Expect(board_str="X.O|...|..O", score=None, win_lead=None, best_moves=set()),
    _Expect(board_str="X.X|...|..O", score=601, win_lead=6, best_moves={"XOX|...|..O"}),
    _Expect(board_str="X.O|.X.|...", score=0, win_lead=0, best_moves={"X.O|.X.|..O"}),
    _Expect(
        board_str="X.O|X.O|.X.", score=-900, win_lead=-9, best_moves={"X.O|X.O|.XO"}
    ),
]


class TttSolverTest(unittest.TestCase):

    def test_cases(self):
        solver = ttt_solver.get_instance()
        results: list[_Expect] = []
        for board_str, expected_result in zip(_BOARDS, _EXPECTED_RESULTS):
            results.append(_Expect(board_str))
            try:
                board = ttt_board.BoardState.from_array(board_str)
            except ttt_board.IllegalBoardState:
                continue
            intel = solver.solve(board)
            # This resets the entered board. The idea is to also check
            # as_string() and canonicalize when the board is interpretable.
            results[-1].board_str = board.as_string()
            results[-1].win_lead = intel.win_lead
            results[-1].score = intel.score
            results[-1].best_moves = {x.as_string() for x in intel.best_plays}
            self.assertEqual(results[-1], expected_result)

        # Print if needed to copy over the test results in this code.
        # print(results)

        # One final check to ensure we didn't skip or miss anything due to logical error.
        self.assertEqual(results, _EXPECTED_RESULTS)


if __name__ == "__main__":
    unittest.main()
