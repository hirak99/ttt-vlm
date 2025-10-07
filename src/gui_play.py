import streamlit as st

from ttt import ttt_evaluator
from ttt import ttt_solver


class StApp:
    def __init__(self) -> None:
        self._board = ttt_solver.BoardState.from_array("...|...|...")
        self._evaluator = ttt_evaluator.TttEvaluator()
        st.title("Tic Tac Toe Evaluator Demo")
        self._ttt_display = st.empty()

    def _eval_and_update(self, text: str):
        try:
            r, c = [int(x) for x in text.split(",")]
            array_orig = self._board.as_array()
            array = array_orig.copy()
            array[(r - 1) * 3 + (c - 1)] = "X" if self._board.x_to_move else "O"
            self._board = ttt_solver.BoardState.from_array(array)
            status, message = self._evaluator.evaluate_move(array_orig, array)
            st.write(message)
            self._board = ttt_solver.BoardState.from_array(array)
            self._save_state()
        except (ValueError, IndexError):
            st.write("Invalid input")
        except ttt_solver.IllegalBoardState:
            st.write("Illegal move")

    def _show_board(self):
        board_out: list[str] = ["```"]
        board_out.append("   1 2 3")
        for index, line in enumerate(self._board.as_string().split("|")):
            line = " ".join(line)
            board_out.append(f"{index+1}  {line}")
        board_out.append("```")
        self._ttt_display.write("\n".join(board_out))

    def _save_state(self):
        print("Saving...")
        st.session_state["app"] = self
        self._show_board()

    def run(self):
        self._ttt_display = st.empty()
        self._show_board()

        who_to_move = "X" if self._board.x_to_move else "O"

        text = st.text_input(
            f"You are playing for '{who_to_move}'. Enter row, column (example 1,2):"
        )
        if text:
            self._eval_and_update(text)


def _main():
    print("Running")
    print(list(st.session_state.keys()))
    app: StApp
    if "app" in st.session_state:
        print("Loading")
        app = st.session_state["app"]
    else:
        print("Creating")
        app = StApp()
    app.run()


if __name__ == "__main__":
    _main()
