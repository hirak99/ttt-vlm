import time

import streamlit as st

# Note: For streamlit codes, a known issue is that relative import at top level is not okay.
# So we cannot say `from .ttt import ...`.
# See: https://discuss.streamlit.io/t/how-to-import-a-module-from-a-parent-directory-in-a-streamlit-project/59016/4
from ttt import ttt_evaluator
from ttt import ttt_solver


class StApp:
    def __init__(self) -> None:
        board_array = ["."] * 9
        self._boards = [ttt_solver.BoardState.from_array(board_array)]

        self._evaluator = ttt_evaluator.TttEvaluator()
        self._text_key = str(time.time())

    @property
    def _board(self) -> ttt_solver.BoardState:
        return self._boards[-1]

    @property
    def _whose_move(self) -> str:
        x_or_o = "X" if self._board.x_to_move else "O"
        return f"`{x_or_o}`"

    def _eval(self, text: str) -> ttt_solver.BoardState | None:
        try:
            r, c = [int(x) for x in text.split(",")]
            if r < 1 or r > 3 or c < 1 or c > 3:
                st.write("Invalid row or column in input.")
                return
            array_orig = self._board.as_array()
            array = array_orig.copy()
            array[(r - 1) * 3 + (c - 1)] = "X" if self._board.x_to_move else "O"
            status, message = self._evaluator.evaluate_move(array_orig, array)
            st.write(f"Evaluation for {self._whose_move} at ({r}, {c}) -")
            st.write(f"- Evaluation: {status.name}")
            st.write(f"- Explanation: {message}")
            return ttt_solver.BoardState.from_array(array)
        except (ValueError, IndexError):
            st.write("Invalid input")
        except ttt_solver.IllegalBoardState:
            st.write("Illegal move")

    def _save_state(self):
        st.session_state["app"] = self

    def run(self):
        st.title("Tic Tac Toe Evaluation Demo")

        st.markdown(
            "\n".join(
                [
                    "- The goal of this demo is to show how TTT moves played an AI will be evaluated and categorized.",
                    "- To use it, enter tic-tac-toe moves, and notice the evaluations on the right as you play.",
                    "- Automated AI response is not implemented (since the goal is to teach), but you can always choose to play the best moves by looking at evaluation.",
                ]
            )
        )

        st.divider()

        if st.button("Reset"):
            st.session_state.clear()
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Move {self._board.move_count + 1}, {self._whose_move} to move.")

            board_out: list[str] = ["```"]
            board_out.append("   1 2 3")
            for index, line in enumerate(self._board.as_string().split("|")):
                line = " ".join(line)
                board_out.append(f"{index+1}  {line}")
            board_out.append("```")
            st.write("\n".join(board_out))

            text = st.text_input(f"Enter move (example 1,2):", key=self._text_key)

        with col2:
            newboard: ttt_solver.BoardState | None = None
            if text:
                newboard = self._eval(text)

        with col1:
            if newboard is not None:
                if st.button("Commit"):
                    self._boards.append(newboard)
                    self._save_state()
                    self._text_key = str(time.time())
                    st.rerun()


def _main():
    app = st.session_state.setdefault("app", StApp())
    app.run()


if __name__ == "__main__":
    _main()
