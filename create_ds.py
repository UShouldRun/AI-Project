from lib.mcts import MCTS, Optional
from lib.connect4 import Connect4
from csv import writer

import numpy as np
import asyncio
import sys

def progress_print(iteration: int, total_items: int) -> None:
    bar_length: int = 50

    progress = (iteration + 1) / total_items
    percent = int(progress * 100)

    filled_length = int(progress * bar_length)
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)

    sys.stdout.write(f"\r[{bar}] {percent}% ({iteration + 1}/{total_items})")
    sys.stdout.flush()

async def generate_ds(s_rollout: int, dt_size: int) -> np.array:
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
        progress_print(i, dt_size)

        game_records: dict = {1: [], 2: []}

        board: Connect4 = Connect4.init_board(6, 7)
        action: Optional[int] = None
        curr_player: int = 1
        value: int = -1

        while value == -1:
            action = await MCTS.mcts(board, Connect4, s_rollout, max_expansion = 7, debug = False)
            game_records[curr_player].append((Connect4.copy(board), action))

            board       = Connect4.play(board, action)
            value       = Connect4.value(board, action)
            curr_player = Connect4.reverse_player(curr_player)

        # print(f"  result: { 'win' if value == 1 and board.player != curr_player else ('draw' if value == .5 else 'loss') }")
        # print(f"  state:")
        # Connect4.print(board)

        result_dict = {}
        for player in (1, 2):
            if value == 1/2:
                result = 1
            elif value == 1 and player == curr_player:
                result = 0
            else:
                result = 2

            moves_arr = np.array(game_records[player], dtype = object)
            result_dict[player] = (moves_arr, result)

        games[i] = result_dict

    print(f"\nSuccessfully generated all {dt_size} games")

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

    with open(filename, mode = "w", newline = "") as csvfile:
        csv_writer = writer(csvfile)
        csv_writer.writerow(["game_id", "player", "board1", "board2", "action", "result"])

        for game_id, game in enumerate(games):
            for player, (moves, result) in game.items():
                for board, action in moves:
                    csv_writer.writerow(
                        [game_id, player, board.board1, board.board2, action, result]
                    )

async def main():
    if len(sys.argv) != 4:
        print("Usage: python create_ds.py <filename.csv: str> <s_rollout: int> <dt_size: int>")
        sys.exit(1)
    filename: str = sys.argv[1]
    s_rollout: int = int(sys.argv[2])
    dt_size: int = int(sys.argv[3])

    games = await generate_ds(s_rollout = s_rollout, dt_size = dt_size)
    create_csv(games, filename = filename)

if __name__ == "__main__":
    asyncio.run(main())
