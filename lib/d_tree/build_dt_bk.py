import numpy as np

from lib.d_tree.node import Node

import time
from joblib import Parallel, delayed

def entropy(data):

    if len(data) == 0:
        return 0

    counts = np.bincount(data)
    probabilities = counts / len(data)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def split_data(X,y,feature,value):
    true_indices = np.where(X[:, feature] <= value )[0]
    false_indices = np.where(X[:, feature] > value)[0]
    true_X, true_y = X[true_indices], y[true_indices]
    false_X, false_y = X[false_indices], y[false_indices]
    return true_X, true_y, false_X, false_y

def information_gain(y, true_y, false_y):
    p = len(true_y) / len(y)
    return entropy(y) - p * entropy(true_y) - (1-p) * entropy(false_y)

def most_common_label(y):
    """Return the most common label in a dataset."""
    unique_labels, counts = np.unique(y, return_counts=True)
    return unique_labels[np.argmax(counts)]

def evaluate_split(X, y, feature, current_entropy):
    """Evaluate the best split for a single feature."""
    best_gain = 0
    best_value = None
    best_sets = None
    
    # Get unique values for the feature
    feature_values = np.sort(np.unique(X[:, feature]))
    
    # For very large datasets, sample threshold values instead of trying all midpoints
    if len(feature_values) > 100:
        # Use percentiles instead of all values
        percentiles = np.linspace(10, 90, 9)  # 10%, 20%, ..., 90%
        feature_values = np.percentile(feature_values, percentiles)
    else:
        # For smaller sets, use midpoints between consecutive values
        feature_values = [(feature_values[i] + feature_values[i+1])/2 
                         for i in range(len(feature_values)-1)]
    
    for value in feature_values:
        true_X, true_y, false_X, false_y = split_data(X, y, feature, value)
        if len(true_y) < 1 or len(false_y) < 1:
            continue
            
        gain = information_gain(y, true_y, false_y)
        
        if gain > best_gain:
            best_gain = gain
            best_value = value
            best_sets = (true_X, true_y, false_X, false_y)
    
    return best_gain, best_value, best_sets, feature

def build_tree(X,y, max_depth=None, min_samples=2, depth=0):

    if len(set(y)) == 1:
        return Node(results=y[0])

    if len(y) < min_samples:  # Too few samples
        return Node(results=most_common_label(y))

    if max_depth is not None and depth >= max_depth:  # Max depth reached
        return Node(results=most_common_label(y))

    #find the best split 

    best_gain = 0 
    best_criteria = None
    best_sets = None
    n_features = X.shape[1]

    for feature in range(n_features):
        feature_values = np.sort(np.unique(X[:, feature])) 

        for i in range(len(feature_values) - 1):
            value = (feature_values[i] + feature_values[i + 1]) / 2

            true_X, true_y, false_X, false_y = split_data(X, y, feature, value)

            if len(true_y) == 0 or len(false_y) == 0:
                continue

            gain = information_gain(y, true_y, false_y)

            if gain>best_gain:
                best_gain=gain 
                best_criteria=(feature,value)
                best_sets= (true_X, true_y, false_X, false_y)

    if best_gain <= 0:
        return Node(results=most_common_label(y))
    
    # Create child nodes
    true_branch = build_tree(best_sets[0], best_sets[1], max_depth, min_samples, depth + 1)
    false_branch = build_tree(best_sets[2], best_sets[3], max_depth, min_samples, depth + 1)
    
    return Node(feature=best_criteria[0], value=best_criteria[1], 
                true_branch=true_branch, false_branch=false_branch)

def predict(tree, sample):
    if tree.results is not None:
        return tree.results
    else:
        branch= tree.false_branch
        if sample[tree.feature] <= tree.value:
            branch=tree.true_branch
        return predict(branch,sample)

def print_tree(node,feature_names=None, indent=""):

    if node.results is not None:
        print(indent + "Leaf:", node.results)
    else:
        feature_name = f"Feature {node.feature}"
        if feature_names is not None:
            feature_name = feature_names[node.feature]
            
        print(indent + f"{feature_name} <= {node.value:.4f}?")
        print(indent + "  ├─ True:")
        print_tree(node.true_branch, feature_names, indent + "  │   ")
        print(indent + "  └─ False:")
        print_tree(node.false_branch, feature_names, indent + "      ")

