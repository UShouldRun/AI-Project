# Connect4 Implementation Documentation

## Overview

This implementation of Connect4 serves as a concrete example of applying the MCTS (Monte Carlo Tree Search) algorithm to a classic board game. Connect4 is implemented as a 2-player game where players alternate placing pieces into a vertical grid, aiming to connect four pieces horizontally, vertically, or diagonally.

## Components

### Connect4Board

The `Connect4Board` class represents the game state using bit boards for efficient representation and operations.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `player` | int | Current player (1 or 2) |
| `rows` | int | Number of rows in the board |
| `cols` | int | Number of columns in the board |
| `board1` | int | Bit board for player 1's pieces |
| `board2` | int | Bit board for player 2's pieces |
| `heights` | List[int] | Current height of each column |

#### Methods

| Method | Description |
|--------|-------------|
| `place_piece(player, row, col)` | Places a piece for the given player at the specified position |
| `get_piece(row, col)` | Returns the player (1 or 2) whose piece is at the given position, or 0 if empty |

### Connect4 (MCTSInterface Implementation)

`Connect4` implements the `MCTSInterface` to provide the game-specific logic for the MCTS algorithm.

#### Methods

##### State Transitions

```python
@staticmethod
def play(state: Connect4Board, action: int) -> Connect4Board
```
Executes a move (placing a piece in a column) and returns the updated board state.

##### Action Generation

```python
@staticmethod
def get_actions(state: Connect4Board) -> List[int]
```
Returns a list of valid columns where a piece can be placed (not full).

##### Terminal State Detection

```python
@staticmethod
def is_terminal_state(state: Connect4Board, action: int) -> bool
```
Checks if the game has ended after the given action. This integrates with MCTS's terminal state tracking feature.

##### Value Evaluation

```python
@staticmethod
def value(state: Connect4Board, action: int, player: Optional[int] = None) -> float
```
Returns:
- `-1` if game is not over
- `0.5` for a draw (represented as `1` in internal implementation, scaled to `0.5`)
- `0` or `1` for a loss or win (relative to the current player or specified player)

This method is critical for MCTS's terminal state detection and propagation, as it identifies winning, losing, and draw states.

##### Heuristic Evaluation

```python
@staticmethod
def heuristic(state: Connect4Board, player: int) -> float
```
Evaluates the board position using pattern recognition:
- Counts sequences of pieces (1, 2, 3, or 4 in a row)
- Assigns different weights based on sequence length and whether opposing pieces block
- Normalizes the result to [0,1]

Used by MCTS rollouts when a terminal state isn't reached but evaluation is needed.

##### Pattern Evaluation

```python
@staticmethod
def evaluate_line(line: List[int], player: int) -> int
```
Evaluates a line of 4 consecutive positions:
- +100 for 4-in-a-row (win)
- +10 for 3-in-a-row with an empty space
- +1 for 2-in-a-row with two empty spaces
- Negative values for opponent's sequences

##### Player Management

```python
@staticmethod
def get_current_player(state: Connect4Board) -> int
```
Returns the current player (1 or 2).

```python
@staticmethod
def reverse_player(player: int) -> int
```
Returns the opponent of the given player.

##### State Manipulation

```python
@staticmethod
def copy(state: Connect4Board) -> Connect4Board
```
Creates a deep copy of the board state.

```python
@staticmethod
def print(state: Connect4Board) -> None
```
Displays the board in a human-readable format.

##### Helper Methods

```python
@staticmethod
def init_board(rows: int, cols: int) -> Connect4Board
```
Creates a new Connect4 board with the specified dimensions.

```python
@staticmethod
def action_get_row(state: Connect4Board, col: int) -> int
```
Determines which row a piece will land in when dropped in a given column.

```python
@staticmethod
def is_out_of_bounds(state: Connect4Board, row: int, col: int) -> bool
```
Checks if a position is outside the board boundaries.

```python
@staticmethod
def check_result(state: Connect4Board, action: int) -> int
```
Checks if the last action resulted in a win, draw, or ongoing game:
- `0` if the game is ongoing
- `1` for a draw
- `2` for a win

This method is used by the `value` method to detect terminal states.

```python
@staticmethod
def count_in_direction(state: Connect4Board, start_row: int, start_col: int, drow: int, dcol: int, player: int) -> int
```
Counts consecutive pieces of a player in a specific direction from a starting position.

## Terminal State Handling

The Connect4 implementation integrates with MCTS's terminal state handling capabilities:

1. The `value` method returns:
   - `-1` for non-terminal states
   - Values in `[0,1]` for terminal states

2. The `check_result` method identifies:
   - Wins by checking for 4-in-a-row in all directions
   - Draws by checking if the top row is full
   - Ongoing games otherwise

This allows MCTS to:
- Quickly identify winning moves
- Avoid losing moves when possible
- Track and store optimal action sequences
- Efficiently prune the search tree

## Performance Considerations

The implementation uses bit boards (`board1` and `board2`) to represent the game state efficiently. This approach:

1. Reduces memory usage
2. Speeds up move generation and validation
3. Makes copying game states faster
4. Enables efficient pattern detection

## Usage with MCTS

To use this Connect4 implementation with the MCTS algorithm:

```python
from lib.mcts import MCTS
from lib.connect4 import Connect4, Connect4Board

# Create initial board
board = Connect4.init_board(6, 7)  # Standard Connect4 dimensions

# Run MCTS to find the best move
best_move = await MCTS.mcts(
    state=board,
    world=Connect4,
    s_rollout=1000,        # Number of simulations
    max_expansion=10,      # Maximum children per node
    s_initial_rollout=100, # Initial random rollouts
    c=1.41,                # Exploration parameter
    heuristic=(True, 20)   # Use heuristic evaluation with max depth 20
)

# Make the move
new_board = Connect4.play(board, best_move)
```

The MCTS algorithm will build a search tree to find the most promising move, using the Connect4 implementation to simulate game play and evaluate positions, with special handling for terminal states to optimize search efficiency.
