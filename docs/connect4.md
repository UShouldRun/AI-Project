# Connect4 Implementation Documentation

Implements Connect4 as an example for MCTS. Two players aim to connect four pieces.

## Components

### `Connect4Board`

Represents the Connect4 game board using bitboards.

#### Attributes

-   `player: int`: Current player (1 or 2).
-   `rows: int`: Number of rows.
-   `cols: int`: Number of columns.
-   `board1: int`: Bitboard for player 1.
-   `board2: int`: Bitboard for player 2.
-   `heights: List[int]`: Current height of each column.

#### Methods

-   `__init__(rows, cols)`: Initializes the board.
-   `place_piece(player, row, col) -> None`: Places a piece.
-   `get_piece(row, col) -> int`: Gets the piece at a position (0 if empty).

### `Connect4` (`MCTSInterface` Implementation)

Implements `MCTSInterface` for Connect4 game logic.

#### Static Methods

-   `play(state: Connect4Board, action: int) -> Connect4Board`: Executes a move.
-   `get_actions(state: Connect4Board) -> List[int]`: Returns valid column actions.
-   `is_terminal_state(state: Connect4Board, action: int) -> bool`: Checks if game ended.
-   `value(state: Connect4Board, action: int, player: Optional[int] = None) -> float`: Returns game value (-1: ongoing, 0.5: draw, 0/1: loss/win).
-   `heuristic(state: Connect4Board, player: int) -> float`: Evaluates board position \[0,1].
-   `evaluate_line(line: List[int], player: int) -> int`: Evaluates a line of 4.
-   `get_current_player(state: Connect4Board) -> int`: Returns current player.
-   `reverse_player(player: int) -> int`: Returns opponent.
-   `copy(state: Connect4Board) -> Connect4Board`: Creates a board copy.
-   `print(state: Connect4Board) -> None`: Prints the board.
-   `init_board(rows: int, cols: int) -> Connect4Board`: Creates a new board.
-   `action_get_row(state: Connect4Board, col: int) -> int`: Gets the row for an action.
-   `is_out_of_bounds(state: Connect4Board, row: int, col: int) -> bool`: Checks board boundaries.
-   `check_result(state: Connect4Board, action: int) -> int`: Checks for win/draw/ongoing (0/1/2).
-   `count_in_direction(state: Connect4Board, row: int, col: int, drow: int, dcol: int, player: int) -> int`: Counts consecutive pieces.

## Performance

Uses bitboards for efficient state representation and operations.

## Usage with MCTS

Example of using `Connect4` with `MCTS`:

```python
from lib.mcts import MCTS
from lib.connect4 import Connect4, Connect4Board
import asyncio

async def main():
    board = Connect4.init_board(6, 7)
    best_move = await MCTS.mcts(
        root_state=board,
        world=Connect4,
        s_rollout=1000,
        s_initial_rollout=100,
        c=1.414,
        max_expansion=7,
        debug=False,
        timer=True
    )
    print(f"Best move: Column {best_move}")
    new_board = Connect4.play(board, best_move)
    Connect4.print(new_board)

if __name__ == "__main__":
    asyncio.run(main())
