import os

from . import llm_generate_data
from ..llm_experiments import llm_generate_data
from ..ttt import ttt_board
from ..ttt import ttt_evaluator

from typing import Iterator

_DATA_PATH = "../../_data"


class DataAnalyzer:
    def __init__(self) -> None:
        self._evaluator = ttt_evaluator.TttEvaluator()

    def analyse(
        self, data: llm_generate_data.PlayerResult
    ) -> tuple[ttt_evaluator.TttEvaluation, str]:
        initial_board = ttt_board.BoardState.from_array(data.board)
        if data.error:
            return ttt_evaluator.TttEvaluation.PARSE_ERROR, data.error

        match data.ai_whose_move:
            case "X" | "O":
                # NOTE: AI is often incorrect in stating whose move it is, but
                # still outputs a valid next move.
                pass
                # ai_x_to_move = data.ai_whose_move == "X"
                # if ai_x_to_move != initial_board.x_to_move:
                #     return (
                #         ttt_evaluator.TttEvaluation.WRONG_PLAYER,
                #         "Incorrectly determined whose move.",
                #     )
            case "ended":
                if initial_board.ended:
                    return (
                        ttt_evaluator.TttEvaluation.BEST_MOVE,
                        "Detected that game has ended.",
                    )
                else:
                    return (
                        ttt_evaluator.TttEvaluation.ASSUMED_ENDED,
                        "Incorrectly determined that game ended.",
                    )
            case _:
                return (
                    ttt_evaluator.TttEvaluation.WRONG_PLAYER,
                    "Invalid entry for ai_whose_move.",
                )

        if data.ai_move is None:
            return ttt_evaluator.TttEvaluation.PARSE_ERROR, "AI move is empty."

        return self._evaluator.evaluate_move(data.board, data.ai_move)


def data_iterator() -> Iterator[llm_generate_data.PlayerResult]:
    for fname in os.listdir(_DATA_PATH):
        with open(os.path.join(_DATA_PATH, fname)) as f:
            for line in f:
                yield llm_generate_data.PlayerResult.model_validate_json(line)
