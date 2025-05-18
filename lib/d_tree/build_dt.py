import numpy as np

from lib.d_tree.node import Node
import pickle

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

def count_nodes(node):
    """Count the number of nodes in a tree."""
    if node.results is not None:  # Leaf node
        return 1
    return 1 + count_nodes(node.true_branch) + count_nodes(node.false_branch)

def collect_leaf_values(node, values=None):
    """
    Collects all leaf values in a subtree.
    """
    if values is None:
        values = []
    
    if node.results is not None:
        values.append(node.results)
    else:
        collect_leaf_values(node.true_branch, values)
        collect_leaf_values(node.false_branch, values)
    
    return values

def most_common_leaf_value(node):
    """
    Find the most common leaf value in a subtree.
    This helps determine what the majority class would be if we pruned.
    """
    leaf_values = collect_leaf_values(node)
    unique_values, counts = np.unique(leaf_values, return_counts=True)
    return unique_values[np.argmax(counts)]

def get_tree_depth(node):
    if node is None or node.results is not None:
        return 0
    return 1 + max(get_tree_depth(node.true_branch), get_tree_depth(node.false_branch))

def evaluate_split_continuous(X, y, feature):
    """Evaluate the best split for a single feature."""
    best_gain = 0
    best_value = None
    best_sets = None
    
    # Get unique values for the feature
    feature_values = np.sort(np.unique(X[:, feature]))
   


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

def evaluate_split(X, y, feature):
    """Evaluate the best split for a single feature."""
    best_gain = 0
    best_value = None
    best_sets = None
    
    # Get unique values for the feature
    feature_values = np.sort(np.unique(X[:, feature]))
   
    n_unique = len(feature_values)
    
    # For turn feature (assuming it's feature 0), use more granular splits
    if feature == 0:
        # For turn number, check more potential splits
        if n_unique > 10:
            # For early, mid and late game splits
            candidate_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            feature_values = np.percentile(feature_values, candidate_percentiles)
        
    # For board position features, focus on meaningful values (empty, player1, player2)
    elif 1 <= feature <= 42:
        # For board positions, just check the midpoints between unique values
        if n_unique > 1:
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

        gain, value, sets, _ = evaluate_split(X,y,feature)
       
        if gain>best_gain:
            best_gain=gain 
            best_criteria=(feature,value)
            best_sets= sets

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

def batch_predict(tree, X, batch_size=1000):
    """Make predictions in batches to save memory."""
    predictions = np.zeros(len(X), dtype=int)
    
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_predictions = np.array([predict(tree, sample) for sample in batch_X])
        predictions[i:i+batch_size] = batch_predictions
        
    return predictions

def save_tree(tree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def tree_to_rules(node, feature_names=None, path=None):
    """Convert a decision tree to a set of rules."""
    if path is None:
        path = []
    
    if node.results is not None:
        # We've reached a leaf node
        return [{"conditions": path.copy(), "prediction": node.results}]
    
    # Format feature name
    feature_name = f"Feature {node.feature}"
    if feature_names is not None:
        feature_name = feature_names[node.feature]
    
    # Rules for left branch (true)
    left_path = path + [(feature_name, "<=", node.value)]
    left_rules = tree_to_rules(node.true_branch, feature_names, left_path)
    
    # Rules for right branch (false)
    right_path = path + [(feature_name, ">", node.value)]
    right_rules = tree_to_rules(node.false_branch, feature_names, right_path)
    
    return left_rules + right_rules

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

