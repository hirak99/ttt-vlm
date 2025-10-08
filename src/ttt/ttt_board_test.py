import unittest

from . import ttt_board

_TEST_BOARDS = [
    "...|...|...",
    "X..|...|...",
    "X..|...|..O",
    "X.X|...|..O",
    "X.O|.X.|...",
    "X.O|X.O|.X.",
]

_ILLEGAL_BOARDS = ["X.O|...|..O"]


class TttSolverTest(unittest.TestCase):

    def test_to_string_from_array(self):
        for board_str in _TEST_BOARDS:
            board = ttt_board.BoardState.from_array(board_str)
            self.assertEqual(board.as_string(), board_str)

    def test_illegal_boards(self):
        for board_str in _ILLEGAL_BOARDS:
            with self.assertRaises(ttt_board.IllegalBoardState):
                ttt_board.BoardState.from_array(board_str)

    def test_x_to_move(self):
        self.assertTrue(ttt_board.BoardState.from_array("...|...|...").x_to_move)
        self.assertTrue(ttt_board.BoardState.from_array(".X.|...|O..").x_to_move)

        self.assertFalse(ttt_board.BoardState.from_array("X..|...|...").x_to_move)
        self.assertFalse(ttt_board.BoardState.from_array(".X.|.X.|O..").x_to_move)


if __name__ == "__main__":
    unittest.main()
