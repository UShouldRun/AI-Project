# Monte Carlo Tree Search (MCTS) Library Documentation

## Overview

This library implements Monte Carlo Tree Search (MCTS), a heuristic search algorithm for decision processes. MCTS combines tree search with random sampling to evaluate actions and build a search tree. It's particularly effective for domains with large branching factors, such as board games.

The implementation uses the UCT (Upper Confidence Bound for Trees) formula for node selection and provides an abstract interface for problem-specific implementations.

## Components

### MCTSInterface (Abstract Base Class)

`MCTSInterface` is an abstract class that must be implemented for any domain-specific problem to use the MCTS algorithm. It defines the required methods for:

- State transitions
- Action generation
- Terminal state detection
- Value evaluation
- Heuristic evaluation
- Player management
- State manipulation and display

#### Required Methods

| Method | Description |
|--------|-------------|
| `play(state, action)` | Executes an action on a state and returns the resulting state |
| `get_actions(state)` | Returns a list of valid actions for the given state |
| `is_terminal_state(state, action)` | Checks if a state is terminal (no further actions possible) |
| `value(state, action, player=None)` | Returns the value of a state (should be in [0,1]) |
| `heuristic(state, player)` | Returns a heuristic evaluation of a state (should be in [0,1]) |
| `get_current_player(state)` | Returns the current player for the given state |
| `reverse_player(player)` | Returns the opponent of the given player |
| `copy(state)` | Creates a deep copy of the given state |
| `print(state)` | Displays the state in a human-readable format |

### MCTSNode

`MCTSNode` represents a node in the MCTS search tree. Each node contains:

- Current state
- Action that led to this state
- Parent node reference
- Child nodes
- Statistics (visits, rewards)
- Terminal state flag

#### Key Methods

| Method | Description |
|--------|-------------|
| `is_root()` | Checks if this node is the root of the tree |
| `is_leaf()` | Checks if this node has no children |
| `has_undetermined_child()` | Checks if any child has unknown terminal status |
| `add_child(child)` | Adds a child node |
| `remove_children()` | Removes all children |
| `get_children()` | Returns an array of child nodes |
| `get_non_terminal_children_idx()` | Returns indices of non-terminal children |
| `get_leafs()` | Returns all leaf nodes in the subtree |

### MCTS (Main Algorithm)

The `MCTS` class implements the core algorithm with the following key steps:

1. **Selection**: Navigate the tree from root to leaf using UCT formula
2. **Expansion**: Generate child nodes for unexplored actions
3. **Simulation**: Perform random rollouts to estimate state value
4. **Backpropagation**: Update statistics back up the tree

#### Main Method

```python
async def mcts(
    state, world, s_rollout, s_initial_rollout=100, 
    c=round(sqrt(2), 3), debug=False, timer=False, 
    heuristic=(False, None)
) -> Action
```

Parameters:
- `state`: Initial state
- `world`: Implementation of MCTSInterface
- `s_rollout`: Number of rollouts to perform
- `s_initial_rollout`: Number of initial random rollouts
- `c`: Exploration parameter for UCT formula (default: √2)
- `debug`: Enable debug output
- `timer`: Enable performance timing
- `heuristic`: Tuple of (use_heuristic, max_depth)

Returns the best action according to the search.

## Helper Methods

The MCTS class includes various helper methods for:

- Node evaluation using UCT formula
- Terminal state detection and propagation
- Converting evaluations between different ranges
- Selecting optimal actions
- Debugging and visualization

## Performance Optimization

The implementation uses NumPy arrays for efficient data storage and manipulation. The code includes timing capabilities to measure the performance of different algorithm phases:

- Selection
- Expansion
- Simulation (rollout)
- Backpropagation

## Usage

To use this library:

1. Implement the `MCTSInterface` for your specific problem domain
2. Create an initial state
3. Call `MCTS.mcts()` with appropriate parameters
4. Apply the returned action to your game/problem

The library handles automatic memory management, efficient tree traversal, and statistical tracking for the search process.
