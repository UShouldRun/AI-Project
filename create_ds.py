from lib.mcts import MCTS, Optional, List
from lib.connect4 import Connect4
from csv import writer
from random import randint, choice
# from memory_profiler import profile

import sys
import os
import gc

def progress_print(iteration: int, total_items: int) -> None:
    """Print a progress bar to show completion status."""
    bar_length: int = 50
    progress = (iteration + 1) / total_items
    percent = int(progress * 100)
    filled_length = int(progress * bar_length)
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)
    sys.stdout.write(f"\r[{bar}] {percent}% ({iteration + 1}/{total_items})")
    sys.stdout.flush()

def generate_and_save_game(s_rollout: int, csv_writer, game_idx: int, total_games: int) -> None:
    """
    Generate a single game and immediately write moves to CSV to minimize memory usage.
    We avoid storing the entire game in memory.
    """
    progress_print(game_idx, total_games)
    
    game_records: dict = { 1: [], 2: [] }
    board: Connect4Board = Connect4.init_board(6, 7)
    curr_player: int = 1
    value: int = -1
    
    j: int = 0
    n: int = randint(0, 8)
    while j < n:
        valid_actions: List[int] = Connect4.get_actions(board)
        if not valid_actions:
            break

        action: int = choice(valid_actions)
        board = Connect4.play(board, action)
        value = Connect4.value(board, action)
        curr_player = Connect4.reverse_player(curr_player)
        j += 1

        if value != -1:
            board = Connect4.init_board(6, 7)
            value = -1
            curr_player = 1
            j = 0
    
    move_count: int = 0
    while value == -1:
        action, _ = MCTS.mcts(board, Connect4, s_rollout, max_expansion = 7, tree = False)
        
        csv_writer.writerow([
            curr_player,
            board.board1, board.board2,
            action
        ])
        
        board = Connect4.play(board, action)
        value = Connect4.value(board, action)
        curr_player = Connect4.reverse_player(curr_player)
        move_count += 1
        
        if move_count % 5 == 0:
            gc.collect(2)
    
    gc.collect(2)

def create_dataset(filename: str, s_rollout: int, dt_size: int):
    """
    Generate Connect4 dataset and write directly to CSV.
    This approach avoids storing games in memory.
    """
    file_exists = os.path.exists(filename)
    mode = 'a' if file_exists else 'w'
    
    with open(filename, mode = mode, newline = "") as csvfile:
        csv_writer = writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["player", "board1", "board2", "action"])
        
        for i in range(dt_size):
            generate_and_save_game(s_rollout, csv_writer, i, dt_size)
            gc.collect(2)
    
    print(f"\nSuccessfully generated and saved {dt_size} games")

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_ds.py <filename.csv: str> <s_rollout: int> <dt_size: int>")
        sys.exit(1)
    
    filename = sys.argv[1]
    s_rollout = int(sys.argv[2])
    dt_size = int(sys.argv[3])

    create_dataset(filename, s_rollout, dt_size)
    gc.collect(2)

if __name__ == "__main__":
    main()
