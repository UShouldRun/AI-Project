import numpy as np
from classes.node import Node
from functools import lru_cache

@lru_cache(maxsize=None)
def _entropy_from_counts(counts_tuple):
    """Compute entropy from a tuple of class counts (cached)."""
    counts = np.array(counts_tuple)
    probs = counts / counts.sum()
    mask = probs > 0
    return -(probs[mask] * np.log2(probs[mask])).sum()


def entropy(y):
    """Compute entropy of label array y."""
    counts = np.bincount(y)
    return _entropy_from_counts(tuple(counts))


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2):
    """
    Build a decision tree using an optimized ID3 algorithm.

    Parameters:
      X: 2D numpy array of features
      y: 1D numpy array of integer labels
      depth: current depth (for stopping)
      max_depth: maximum allowed depth (None for unlimited)
      min_samples_split: minimum samples to split a node

    Returns:
      Node: root of the decision tree
    """
    # Stopping conditions
    if len(y) < min_samples_split or len(set(y)) == 1:
        # Return leaf with majority class
        majority = np.bincount(y).argmax()
        return Node(results=majority)
    if max_depth is not None and depth >= max_depth:
        majority = np.bincount(y).argmax()
        return Node(results=majority)

    current_entropy = entropy(y)
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    n_features = X.shape[1]
    for feature in range(n_features):
        # Use numpy.unique (C-optimized) instead of Python set
        values = np.unique(X[:, feature])
        for value in values:
            mask = X[:, feature] <= value
            true_y = y[mask]
            false_y = y[~mask]
            # Skip if no split
            if len(true_y) == 0 or len(false_y) == 0:
                continue

            p = len(true_y) / len(y)
            gain = current_entropy - p * entropy(true_y) - (1 - p) * entropy(false_y)
            if gain > best_gain:
                best_gain = gain
                best_criteria = (feature, value)
                best_sets = (
                    X[mask], true_y,
                    X[~mask], false_y
                )

    # If no gain, return leaf
    if best_gain <= 0:
        majority = np.bincount(y).argmax()
        return Node(results=majority)

    # Recurse on branches
    true_branch = build_tree(
        best_sets[0], best_sets[1],
        depth + 1, max_depth, min_samples_split
    )
    false_branch = build_tree(
        best_sets[2], best_sets[3],
        depth + 1, max_depth, min_samples_split
    )
    return Node(
        feature=best_criteria[0],
        value=best_criteria[1],
        true_branch=true_branch,
        false_branch=false_branch
    )


def predict(tree, sample):
    """Predict class label for a single sample using the tree."""
    if tree.results is not None:
        return tree.results
    branch = (tree.true_branch if sample[tree.feature] <= tree.value
              else tree.false_branch)
    return predict(branch, sample)


def print_tree(node, indent=""):
    """Recursively prints the decision tree structure."""
    if node.results is not None:
        print(f"{indent}Leaf: {node.results}")
    else:
        print(f"{indent}Feature {node.feature} <= {node.value}?")
        print(f"{indent}  ├─ True:")
        print_tree(node.true_branch, indent + "  │   ")
        print(f"{indent}  └─ False:")
        print_tree(node.false_branch, indent + "      ")
