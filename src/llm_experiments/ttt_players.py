import abc
import json
import logging
import random

from ..llm_service import llm
from ..ttt import ttt_board

# This should be whose_move if AI thinks the game ended.
_ENDED = "ended"


class AbstractPlayer(abc.ABC):
    @abc.abstractmethod
    def play(self, board: ttt_board.BoardState) -> str:
        pass

    @abc.abstractmethod
    def model_description(self) -> str:
        pass


class LlmPlayer(AbstractPlayer):
    def __init__(self, model_name: str) -> None:
        self._llm = llm.OpenAiLlmInstance(model_name)

    def play(self, board: ttt_board.BoardState) -> str:
        prompt_lines = [
            "Given this tic-tac-toe board, please state your next move.",
            f"{json.dumps(board.as_array())}",
            "---",
            "Your output must be a valid json of the following format -",
            "{",
            f'  "whose_move": "X" or "O" or "{_ENDED}",',
            '  "updated_board": [UPDATED_BOARD_AFTER_MOVE]',
            "}",
            'Omit "updated_board" if game has ended.',
        ]
        prompt = "\n".join(prompt_lines)
        logging.info(f"Prompt: {prompt}")
        return self._llm.do_prompt(prompt, max_tokens=1024)

    def model_description(self) -> str:
        return self._llm.model_description()


class RandomPlayer(AbstractPlayer):
    def play(self, board: ttt_board.BoardState) -> str:
        # Randomly select who to move.
        whose_move = random.choice(["X", "O"])
        # Random move to one of the empty squares.
        updated_board = board.as_array()
        empty_indices = [i for i, x in enumerate(updated_board) if x == "."]
        if empty_indices:
            updated_board[random.choice(empty_indices)] = whose_move
        else:
            whose_move = _ENDED
        return json.dumps(
            {
                "whose_move": whose_move,
                "updated_board": updated_board,
            }
        )

    def model_description(self) -> str:
        return "Random"


def player_factory(player_name: str) -> AbstractPlayer:
    if player_name == "random":
        return RandomPlayer()
    elif player_name.startswith("gpt") or player_name.startswith("o3"):
        return LlmPlayer(player_name)
    else:
        raise ValueError(f"Unknown player name: {player_name}")
