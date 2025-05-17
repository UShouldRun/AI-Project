from csv import writer
from random import randint, choice
import sys
import os
import gc

# Import either pure Python or Cython version based on availability
try:
    from lib.mcts.mcts_cython import MCTS  # Cython version
except ImportError:
    from lib.mcts import MCTS  # Fallback to Python version
    print("Using pure Python MCTS (Cython not available)")

from lib.connect4 import Connect4
from typing import Optional, List

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
    progress_print(game_idx, total_games)
    
    game_records = {1: [], 2: []}
    board = Connect4.init_board(6, 7)
    curr_player = 1
    value = -1
    
    # Random opening moves
    j = 0
    n = randint(0, 12)
    while j < n:
        valid_actions = [int(a) for a in Connect4.get_actions(board)]  # Ensure actions are int
        if not valid_actions:
            break

        action = int(choice(valid_actions))  # Ensure action is int
        board = Connect4.play(board, action)
        value = int(Connect4.value(board, action))  # Ensure value is int
        curr_player = int(Connect4.reverse_player(curr_player))  # Ensure player is int
        j += 1

        if value != -1:
            board = Connect4.init_board(6, 7)
            value = -1
            curr_player = 1
            j = 0
    
    # Main game loop
    move_count = 0
    while value == -1:
        if randint(0, 99) < 25:
            # 25% chance of random move for exploration
            valid_actions = [int(a) for a in Connect4.get_actions(board)]
            action = int(choice(valid_actions))
        else:
            # Use MCTS for the move
            action, _ = MCTS.mcts(
                board,  
                Connect4, 
                int(s_rollout),  # Ensure s_rollout is int
                max_expansion=7, 
                tree=False
            )
            action = int(action)  # Ensure action is int
            
            # Write the move to CSV
            csv_writer.writerow([
                curr_player,
                int(board.board1),  # Ensure board values are int
                int(board.board2),
                int(action)
            ])
        
        board = Connect4.play(board, action)
        value = int(Connect4.value(board, action))
        curr_player = int(Connect4.reverse_player(curr_player))
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
    
    with open(filename, mode=mode, newline="") as csvfile:
        csv_writer = writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["player", "board1", "board2", "action"])
        
        for i in range(dt_size):
            generate_and_save_game(s_rollout, csv_writer, i, dt_size)
            gc.collect(2)
    
    print(f"\nSuccessfully generated and saved {dt_size} games")

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_ds.py <filename.csv> <s_rollout> <dt_size>")
        print("Arguments:")
        print("  filename.csv: Output CSV file path")
        print("  s_rollout:    Number of MCTS simulations per move (integer)")
        print("  dt_size:      Number of games to generate (integer)")
        sys.exit(1)
    
    try:
        filename = sys.argv[1]
        s_rollout = int(sys.argv[2])  # Handles both '100' and '100.0'
        dt_size = int(sys.argv[3])
        
        if s_rollout <= 0 or dt_size <= 0:
            raise ValueError("s_rollout and dt_size must be positive integers")

        create_dataset(filename, s_rollout, dt_size)
        gc.collect(2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()