from lib.mcts import MCTS, Optional
from lib.connect4 import Connect4
from csv import writer
from random import randint, choice
from itertools import repeat

import numpy as np
import sys
import os

def progress_print(iteration: int, total_items: int) -> None:
    bar_length: int = 50

    progress = (iteration + 1) / total_items
    percent = int(progress * 100)

    filled_length = int(progress * bar_length)
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)

    sys.stdout.write(f"\r[{bar}] {percent}% ({iteration + 1}/{total_items})")
    sys.stdout.flush()

def generate_ds(s_rollout: int, dt_size: int, total: int, iter: int) -> np.array:
    """
    Generate a dataset of self-play Connect4 games using MCTS.

    Returns:
      np.ndarray of length dt_size, dtype=object. Each element is a dict:
        {
          1: (np.ndarray of (board, action) tuples, result),
          2: (np.ndarray of (board, action) tuples, result)
        }
      where result is 0 (lose), 1 (draw), or 2 (win) for that player.
    """
    games: np.array = np.empty(dt_size, dtype = object)

    for i in range(dt_size):
        progress_print(iter + i, total)

        game_records: dict = {1: [], 2: []}

        board: Connect4 = Connect4.init_board(6, 7)
        action: Optional[int] = None
        curr_player: int = 1
        value: int = -1

        j: int = 0
        n: int = randint(0,8)
        while j < n:
            action = choice(Connect4.get_actions(board))
            board = Connect4.play(board, action)
            value = Connect4.value(board, action)
            j += 1
            if value != -1:
                board = Connect4.init_board(6, 7)
                value = -1
                j     = 0

        while value == -1:
            action, _ = MCTS.mcts(board, Connect4, s_rollout, max_expansion = 7, debug = False)
            game_records[curr_player].append((Connect4.copy(board), action))

            board       = Connect4.play(board, action)
            value       = Connect4.value(board, action)
            curr_player = Connect4.reverse_player(curr_player)

        result_dict = {}
        result_dict[1] = np.array(game_records[1], dtype = object)
        result_dict[2] = np.array(game_records[2], dtype = object)

        games[i] = result_dict

    if iter + dt_size >= total:
        print(f"\nSuccessfully generated all {total} games")

    return games

def create_csv(games: np.array, filename: str):
    """
    Template function to save generated games to a CSV file.

    Each row corresponds to a single move:
      - game_id: index of the game
      - player: 1 or 2
      - board: serialized board state (e.g., list or string)
      - action: column index played
      - result: 0 (lose), 1 (draw), or 2 (win)
    """

    file_exists: bool = os.path.exists(filename)
    mode: str = 'a' if file_exists else 'w'
    with open(filename, mode = mode, newline = "") as csvfile:
        csv_writer = writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["player", "board1", "board2", "action"])

        for game in games:
            for player, moves in game.items():
                for board, action in moves:
                    csv_writer.writerow([
                        player,
                        board.board1, board.board2,
                        action
                    ])

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_ds.py <filename.csv: str> <s_rollout: int> <dt_size: int>")
        sys.exit(1)
    filename: str  = sys.argv[1]
    s_rollout: int = int(sys.argv[2])
    dt_size: int   = int(sys.argv[3])

    n: int = dt_size // 5
    remainder: int = dt_size % 5
    for i in range(n):
        games = generate_ds(s_rollout = s_rollout, dt_size = 5, total = dt_size, iter = 5 * i)
        create_csv(games, filename = filename)
    if remainder > 0:
        games = generate_ds(s_rollout = s_rollout, dt_size = remainder, total = dt_size, iter = n)
        create_csv(games, filename = filename)

if __name__ == "__main__":
    main()
