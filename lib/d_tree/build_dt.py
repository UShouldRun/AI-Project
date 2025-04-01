import numpy as np

from classes.node import Node

def entropy(data):
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

def build_tree(X,y):
    if len(set(y)) == 1:
        return Node(results=y[0])

    best_gain = 0 
    best_criteria = None
    best_sets = None
    n_features = X.shape[1]

    current_entropy=entropy(y)
    
    for feature in range(n_features):
        feature_values=set(X[:, feature])
        
        for value in feature_values:
            true_X, true_y, false_X, false_y = split_data(X, y, feature, value)
            true_entropy = entropy(true_y)
            false_entropy = entropy(false_y)
            p=len(true_y)/len(y)
            gain= current_entropy -p * true_entropy - (1-p) * false_entropy
            
            if gain>best_gain:
                best_gain=gain 
                best_criteria=(feature,value)
                best_sets= (true_X, true_y, false_X, false_y)

        if best_gain>0:
            true_branch = build_tree(best_sets[0], best_sets[1])
            false_branch = build_tree(best_sets[2], best_sets[3])
            node= Node(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)
            return node
        return Node(results=y[0])

def predict(tree, sample):
    if tree.results is not None:
        return tree.results
    else:
        branch= tree.false_branch
        if sample[tree.feature] <= tree.value:
            branch=tree.true_branch
        return predict(branch,sample)

def print_tree(node, indent=""):
    """ Recursively prints the decision tree. """
    if node.results is not None:
        print(indent + "Leaf:", node.results)
    else:
        print(indent + f"Feature {node.feature} <= {node.value}?")
        print(indent + "  ├─ True:")
        print_tree(node.true_branch, indent + "  │   ")
        print(indent + "  └─ False:")
        print_tree(node.false_branch, indent + "      ")
