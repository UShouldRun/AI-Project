import numpy as np
from lib.mcts import MCTS
from lib.connect4 import Connect4
from csv import writer
import asyncio


async def generate_ds(s_rollout: int, dt_size: int):
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
    games = np.empty(dt_size, dtype=object)

    for i in range(dt_size):
        board = Connect4.init_board(6, 7)
        game_records = {1: [], 2: []}
        action = None
        current_player = 1

        # Play until terminal state
        while not Connect4.is_terminal_state(board, action):
            action = await MCTS.mcts(board, Connect4, s_rollout, max_expansion=7)
            game_records[current_player].append((Connect4.copy(board), action))
            board = Connect4.play(board, action)
            current_player = Connect4.reverse_player(current_player)

        # Determine game outcome: 0=draw, 1 or 2 for the winner
        winner = Connect4.value(board)

        # Build the per-player entries
        result_dict = {}
        for player in (1, 2):
            if winner == 1/2:
                result = 1  # draw
            elif winner == 1 and player == current_player:
                result = 0  # win
            else:
                result = 2  # lose

            # convert move list to numpy array of object tuples
            moves_arr = np.array(game_records[player], dtype=object)
            result_dict[player] = (moves_arr, result)

        games[i] = result_dict

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

    with open(filename, mode="w", newline="") as csvfile:
        writer = writer(csvfile)
        writer.writerow(["game_id", "player", "board1", "board2", "action", "result"])
        # Iterate over games and write each move
        for game_id, game in enumerate(games):
            for player, (moves, result) in game.items():
                for board, action in moves:
                    writer.writerow([game_id, player, board.board1, board.board2, action, result])

async def main():
    games = await generate_ds(s_rollout=1000, dt_size=10)
    create_csv(games, filename="dt.csv")

if __name__ == "__main__":
    asyncio.run(main())