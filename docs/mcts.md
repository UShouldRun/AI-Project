# mcts.py Documentation

Implements the Monte Carlo Tree Search (MCTS) algorithm for decision-making in games.

## Overview

MCTS simulates playouts to build a search tree, balancing exploration and exploitation to find optimal actions. Key components:

-   **`MCTSInterface`:** Abstract base class defining the game/environment interaction contract.
-   **`MCTSNode`:** Represents a node in the search tree, storing action, parent, children, visits, reward, and terminal status.
-   **`MCTS`:** Class containing static methods for the MCTS algorithm (selection, expansion, rollout, backpropagation).

## `MCTSInterface`

Abstract class for game/environment interaction. Subclasses must implement these static methods:

-   `play(state: State, action: Action) -> State`: Executes an action on a state.
-   `get_actions(state: State) -> List[Action]`: Returns valid actions for a state.
-   `is_terminal_state(state: State, action: Action) -> bool`: Checks if a state is terminal.
-   `value(state: State, action: Action, player: Optional[int] = None) -> float`: Returns the state's value \[0,1].
-   `heuristic(state: State, player: int) -> float`: Returns a heuristic value of the state \[0,1].
-   `get_current_player(state: int) -> int`: Returns the current player.
-   `reverse_player(player: int) -> int`: Returns the opponent.
-   `copy(state: State) -> State`: Creates a state copy.
-   `print(state: State) -> None`: Prints the state.

## `MCTSNode`

Represents a node in the MCTS tree.

### Attributes

-   `action: Action`: Action leading to this state.
-   `parent: Optional["MCTSNode"]`: Parent node.
-   `children: np.ndarray`: Array of child nodes.
-   `s_children: int`: Number of actual children.
-   `max_children: int`: Maximum allowed children.
-   `depth: int`: Node depth.
-   `reward: float`: Accumulated reward.
-   `visits: int`: Number of visits.
-   `terminal: float`: Terminal status (-1: no, 0: loss, 0.5: draw, 1: win).
-   `undet_children: int`: Number of unexplored children.

### Methods

-   `__init__(action, parent, max_children)`: Initializes a new node.
-   `is_root() -> bool`: Checks if it's the root.
-   `is_leaf() -> bool`: Checks if it has no children.
-   `has_undetermined_child() -> bool`: Checks for unexplored children.
-   `add_child(child: "MCTSNode") -> None`: Adds a child.
-   `remove_children() -> None`: Removes all children.
-   `get_children() -> np.ndarray`: Returns actual children.

## `MCTS`

Static methods implementing the MCTS algorithm.

-   `_encapsulate(action, max_expansion) -> MCTSNode`: Creates a root node.
-   `_is_terminal_node(node) -> bool`: Checks if a node is terminal.
-   `_inverse_sigmoid(x) -> float`: Computes inverse sigmoid.
-   `_convert_eval(value) -> float`: Converts \[0,1] to real line.
-   `_convert_terminal(value) -> float`: Converts terminal value to +/- inf.
-   `_sort_children(root) -> None`: Sorts children by evaluation.
-   `_pick_action(root) -> Action`: Returns the best action (first child).
-   `_only_action(node, state, world, c) -> Optional[Action]`: Checks for a single non-losing action.
-   `_random_rollout(root, root_state, world, n=100) -> None`: Performs random rollouts.
-   `_evaluate(node, c) -> float`: Evaluates node using UCB1.
-   `_select(root, root_state, world, c) -> tuple[Optional[MCTSNode], State]`: Selects best child using UCB1.
-   `_expand(node, state, world) -> bool`: Expands a leaf node.
-   `_rollout(leaf, state, world) -> None`: Performs a random playout.
-   `_backpropagate(node, reward) -> None`: Backpropagates reward.
-   `_backpropagate_terminal(node, terminal) -> None`: Backpropagates terminal value.
-   `_print_node(node, state, world, c) -> None`: Prints node information (debugging).
-   `async def mcts(...) -> Action`: Main MCTS function to find the best action.
