# Connect Four with MCTS and Decision Tree AI

An implementation of Connect Four featuring AI players powered by **Monte Carlo Tree Search (MCTS)** and **Decision Trees**.

---

## Overview

This project offers three gameplay modes:
- **Human vs Human**
- **Human vs AI**
- **AI vs AI**

It includes two AI strategies:
- **Monte Carlo Tree Search (MCTS):** A simulation-based search algorithm.
- **Decision Tree AI:** A machine learning model trained on MCTS-generated gameplay data.

For detailed documentation, refer to the files in the [`docs/`](./docs) directory.

---

## Features

- Interactive command-line interface.
- Efficient bitboard-based game state representation.
- Configurable MCTS parameters.
- Decision Tree AI trained on self-play data.
- AI comparison mode for testing strategies.

---

## Requirements

- Python 3.10+
- [NumPy](https://numpy.org/)

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/UShouldRun/AI-Project.git
    cd AI-Project
    ```

2. **Install dependencies:**
    ```bash
    pip install numpy
    ```

3. **Ensure Python 3 is installed:**
    - **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update
        sudo apt install python3 python3-pip
        ```
    - **macOS:** (using Homebrew)
        ```bash
        brew install python3
        ```
    - **Windows:**  
        Download from [python.org](https://www.python.org/) and follow installation instructions.  
        ➔ **Important:** Check "Add Python to PATH" during installation.

---

## Usage

### Running the Game
```bash
python main.py
```

This launches the interactive game interface.

---

### Generating Training Data for the Decision Tree
```bash
python create_ds.py <filename.csv> <mcts_rollouts> <num_games>
```

**Example:**
```bash
python create_ds.py dt_data.csv 1000 5000
```

See [`docs/create_ds.md`](./docs/create_ds.md) for more information.

---

### Training the Decision Tree Model
```bash
python3 lib/d_tree/read_dataset.py
```

More details in [`docs/d_tree.md`](./docs/d_tree.md) (if available).

---

### Playing with Optimizations (AI Mode)
```bash
python3 -O main.py
```

---

## Algorithms

### Monte Carlo Tree Search (MCTS)
- Implemented with UCT (Upper Confidence Bound applied to Trees) for node selection.
- See [`docs/mcts.md`](./docs/mcts.md) for technical details.

### Decision Tree
- Trained on MCTS gameplay data.
- Uses the ID3 algorithm.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments

- Developed as part of the **Artificial Intelligence 2024/2025** course assignment.

---

## Authors

- António Lanção
- Henrique Teixeira
- João Ferreira
