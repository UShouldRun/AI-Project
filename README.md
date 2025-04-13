# Connect Four with MCTS and Decision Tree AI

An implementation of the Connect Four game with multiple AI strategies using Monte Carlo Tree Search (MCTS) and Decision Trees.

## Overview

This project implements the classic Connect Four game with three gameplay modes:
- Human vs Human
- Human vs AI
- AI vs AI (comparing different algorithms)

Two AI approaches are implemented:
1. **Monte Carlo Tree Search (MCTS)** - A probabilistic search algorithm that builds a search tree through random sampling
2. **Decision Tree** - A machine learning model trained on MCTS gameplay data

## Features

- Interactive command-line interface for playing Connect Four
- Efficient bit-board representation of the game state
- Customizable MCTS parameters (exploration constant, rollout depth, etc.)
- Decision Tree model trained on MCTS gameplay data
- Performance metrics and visualizations for AI comparison

## Requirements

- Python 3.10+
- NumPy
- Asyncio

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/connect-four-ai.git
cd connect-four-ai
pip install -r requirements.txt
```

## Usage

### Running the Game

```bash
python main.py
```
This will launch the game interface where you can select the game mode and AI settings.

## Algorithms

### Monte Carlo Tree Search (MCTS)

Our MCTS implementation uses the UCT (Upper Confidence Bound for Trees) formula for node selection. The algorithm follows four main steps:
1. Selection - Navigate the tree from root to leaf
2. Expansion - Generate child nodes for unexplored actions
3. Simulation - Perform random rollouts to estimate value
4. Backpropagation - Update statistics back up the tree

Configuration parameters:
- `c`: Exploration parameter (default: √2)
- `s_rollout`: Number of rollouts to perform
- `heuristic`: Whether to use a heuristic evaluation function

### Decision Tree

The Decision Tree AI is implemented using the ID3 algorithm and trained on data generated from MCTS gameplay. The implementation:
1. Uses MCTS to generate a dataset of (state, best_move) pairs
2. Processes the dataset to extract meaningful features
3. Builds a decision tree using the ID3 algorithm
4. Uses the tree to make move decisions during gameplay

## Training the Decision Tree

To generate training data and build the decision tree:

```bash
python lib/d_tree/build_dt.py --games 1000 --mcts-rollouts 500
```

This will:
1. Play 1000 self-play games using MCTS (with 500 rollouts per move)
2. Generate a dataset of board states and optimal moves
3. Train a decision tree on this dataset
4. Save the model for later use

## Testing

The decision tree implementation was initially tested on the Iris dataset before being applied to the Connect Four domain.

```bash
# Run tests on the Iris dataset
python lib/d_tree/read_iris.py
```

## Performance

The MCTS algorithm's performance scales with the number of rollouts. The decision tree offers faster move computation at the cost of some strategic depth.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of the Artificial Intelligence 2024/2025 course assignment
- Based on concepts from "Adversarial search strategies and Decision Trees"

## Authors

- António Lanção
- Henrique Teixeira
- João Ferreira
