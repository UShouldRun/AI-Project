import numpy as np
import pandas as pd
import os
import time
from lib.d_tree.node import Node
import copy
import matplotlib.pyplot as plt
from lib.d_tree.build_dt import (build_tree, predict, batch_predict, save_tree, load_tree, get_tree_depth,
    count_nodes, most_common_leaf_value)

def train_test_split(X,y, test_ratio=0.2, seed=70):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size= int(len(X)*test_ratio)
    test_indices= indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def train_test_val_split(X, y, test_ratio=0.2, val_ratio=0.2, seed=70):
    """Split data into training, validation and test sets."""
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_ratio)
    val_size = int(len(X) * val_ratio)
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def read_dt_csv(filename:str):
    str_path="utils"
    file_path = os.path.join(str_path, filename)

    df = pd.read_csv(file_path)
    
    feature_names = df.columns[:-1].tolist()

    data = df.to_numpy()

    return data, feature_names

def train_tree(data, feature_names=None, max_depth=None, min_samples=2, test_ratio=0.2):
    X = data[:, :-1]  
    y = data[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=test_ratio)
    decision_tree=build_tree(X_train,y_train, max_depth=max_depth, min_samples=min_samples)

    predictions = [predict(decision_tree, sample) for sample in X_test]
    
    correct = np.sum(predictions == y_test)  
    accuracy = correct / len(y_test) 

    return decision_tree, accuracy

def hyperparameter_tuning(data):

    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Define parameter grid
    max_depths = [None, 7, 10, 20]
    min_samples = [2, 20, 50, 100]
    
    best_score = 0
    best_params = None
    
    print("\nHyperparameter Tuning:")
    print("=====================")
    
    for depth in max_depths:
        for samples in min_samples:
            avg_acc, _ = k_fold(data, k=10, max_depth=depth, min_samples=samples)
            
            if avg_acc > best_score:
                best_score = avg_acc
                best_params = (depth, samples)
    
    print("Best parameters:")
    print(f"max_depth={best_params[0]}, min_samples={best_params[1]}")
    print(f"Best cross-validation accuracy: {best_score:.2%} \n")
    
    return best_params

def k_fold(data, k=10, seed=70, max_depth=None, min_samples=2):

    X = data[:, :-1]  
    y = data[:, -1].astype(int)

    np.random.seed(seed)
    indices= np.arange(len(X))
    np.random.shuffle(indices)

    fold_size=len(X)//k 
    accuracies = []

    for i in range(k):
        start= i*fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(X)

        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples= min_samples)
        
        predictions = [predict(tree, sample) for sample in X_test]
        acc = np.mean(predictions == y_test)
        accuracies.append(acc)

    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    return avg_acc, std_acc


def evaluate_model(tree, X_test, y_test, batch_size=1000):
    """Evaluate model performance."""
    start_time = time.time()
    
    # Make predictions in batches
    y_pred = batch_predict(tree, X_test, batch_size)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    
    prediction_time = time.time() - start_time
    prediction_speed = len(X_test) / prediction_time
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Prediction speed: {prediction_speed:.1f} examples/second")
    
    return accuracy, prediction_time

def evaluate_accuracy(tree, X_test, y_test, batch_size=1000):
    y_pred = batch_predict(tree, X_test, batch_size)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)

    return accuracy

def prune_tree(node, X_val, y_val):
    """
    Prune a decision tree using reduced error pruning.
    
    Args:
        node: The current node being considered for pruning
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        The pruned tree
    """
    # Base case: If we're at a leaf, don't prune
    if node.results is not None:
        return node
    
    # Recursively prune the children
    node.true_branch = prune_tree(node.true_branch, X_val, y_val)
    node.false_branch = prune_tree(node.false_branch, X_val, y_val)
    
    # If both children are leaves, consider pruning this node
    if (node.true_branch.results is not None and 
        node.false_branch.results is not None):
        
        # Measure accuracy before pruning
        accuracy_before = evaluate_accuracy(node, X_val, y_val)
        
        # Create a new leaf node with the majority class
        majority_class = most_common_leaf_value(node)
        temp_node = Node(results=majority_class)
        
        # Measure accuracy after pruning
        accuracy_after = evaluate_accuracy(temp_node, X_val, y_val)
        
        # If pruning improves or maintains accuracy, perform the pruning
        if accuracy_after >= accuracy_before:
            return temp_node
    
    return node

def cost_complexity_pruning(tree, X_val, y_val, alpha=0.01):
    """
    Cost complexity pruning (also known as weakest link pruning).
    
    Args:
        tree: The decision tree to prune
        X_val: Validation features
        y_val: Validation labels
        alpha: Complexity parameter - higher values lead to smaller trees
        
    Returns:
        The pruned tree
    """
    # If we're at a leaf, return as is
    if tree.results is not None:
        return tree
    
    # Recursively prune children
    tree.true_branch = cost_complexity_pruning(tree.true_branch, X_val, y_val, alpha)
    tree.false_branch = cost_complexity_pruning(tree.false_branch, X_val, y_val, alpha)
    
    # If both children are leaves, consider pruning this node
    if (tree.true_branch.results is not None and 
        tree.false_branch.results is not None):
        
        # Calculate error with current subtree
        error_before = 1.0 - evaluate_accuracy(tree, X_val, y_val)
        n_leaves_before = count_nodes(tree)
        
        # Calculate error if we prune this node
        majority_class = most_common_leaf_value(tree)
        pruned_node = Node(results=majority_class)
        error_after = 1.0 - evaluate_accuracy(pruned_node, X_val, y_val)
        n_leaves_after = 1  # Just one leaf if pruned
        
        # Cost complexity measure: error + alpha * number of leaves
        cost_before = error_before + alpha * n_leaves_before
        cost_after = error_after + alpha * n_leaves_after
        
        # If pruning reduces cost complexity, prune
        if cost_after <= cost_before:
            return pruned_node
    
    return tree

def compare_pruning_methods(data):
    """Compare different pruning methods."""
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X, y)
    
    # Build the tree without pruning
    print("\n=== Original Tree ===")
    tree = build_tree(X_train, y_train)
    nodes_count = count_nodes(tree)
    depth = get_tree_depth(tree) 
    train_acc = evaluate_accuracy(tree, X_train, y_train)
    val_acc = evaluate_accuracy(tree, X_val, y_val)
    test_acc = evaluate_accuracy(tree, X_test, y_test)
    
    print(f"Number of nodes: {nodes_count}")
    print(f"Tree depth: {depth}")
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print("\nTree structure:")
    
    # Reduced Error Pruning
    print("\n=== Tree with Reduced Error Pruning ===")
    pruned_tree = prune_tree(copy.deepcopy(tree), X_val, y_val)
    nodes_count = count_nodes(pruned_tree)
    depth = get_tree_depth(pruned_tree) 
    train_acc = evaluate_accuracy(pruned_tree, X_train, y_train)
    val_acc = evaluate_accuracy(pruned_tree, X_val, y_val)
    test_acc = evaluate_accuracy(pruned_tree, X_test, y_test)
    
    print(f"Number of nodes: {nodes_count}")
    print(f"Tree depth: {depth}")
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print("\nTree structure:")
    
    # Cost Complexity Pruning with different alpha values
    alpha_values = [0.001, 0.01, 0.05]
    best_alpha = 0
    best_accuracy = 0
    best_tree = None
    
    print("\n=== Cost Complexity Pruning ===")
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        ccp_tree = cost_complexity_pruning(copy.deepcopy(tree), X_val, y_val, alpha)
        nodes_count = count_nodes(ccp_tree)
        depth = get_tree_depth(ccp_tree)
        val_acc = evaluate_accuracy(ccp_tree, X_val, y_val)
        
        print(f"Number of nodes: {nodes_count}")
        print(f"Tree depth: {depth}")
        print(f"Validation accuracy: {val_acc:.2%}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_alpha = alpha
            best_tree = ccp_tree
    
    # Evaluate best CCP tree
    print("\n=== Best Cost Complexity Pruning Tree ===")
    print(f"Best alpha: {best_alpha}")
    nodes_count = count_nodes(best_tree)
    depth = get_tree_depth(best_tree) 
    train_acc = evaluate_accuracy(best_tree, X_train, y_train)
    val_acc = evaluate_accuracy(best_tree, X_val, y_val)
    test_acc = evaluate_accuracy(best_tree, X_test, y_test)
    
    print(f"Number of nodes: {nodes_count}")
    print(f"Tree depth: {depth}")
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print("\nTree structure:")
    

def k_fold_with_pruning(data, k=5, seed=70):
    """Perform k-fold cross-validation with pruning."""
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    
    # Results tracking
    unpruned_accuracies = []
    pruned_accuracies = []
    ccp_accuracies = []
    node_counts_unpruned = []
    node_counts_pruned = []
    node_counts_ccp = []
    depths_unpruned = []
    depths_pruned = []
    depths_ccp = []
    
    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")
        # Split data into train/val/test
        start_test = i * fold_size
        end_test = (i + 1) * fold_size if i != k - 1 else len(X)
        
        test_indices = indices[start_test:end_test]
        non_test_indices = np.concatenate([indices[:start_test], indices[end_test:]])
        
        # Further split non-test data into train and validation
        np.random.shuffle(non_test_indices)
        val_size = len(non_test_indices) // 5  # Use 20% of remaining data for validation
        val_indices = non_test_indices[:val_size]
        train_indices = non_test_indices[val_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Build unpruned tree
        tree = build_tree(X_train, y_train)
        acc_unpruned = evaluate_accuracy(tree, X_test, y_test)
        nodes_unpruned = count_nodes(tree)
        depth_unpruned = get_tree_depth(tree)
        
        # Reduced Error Pruning
        rep_tree = prune_tree(copy.deepcopy(tree), X_val, y_val)
        acc_pruned = evaluate_accuracy(rep_tree, X_test, y_test)
        nodes_pruned = count_nodes(rep_tree)
        depth_pruned = get_tree_depth(rep_tree)
        
        # Cost Complexity Pruning (CCP) with fixed alpha for simplicity
        ccp_tree = cost_complexity_pruning(copy.deepcopy(tree), X_val, y_val, alpha=0.001)
        acc_ccp = evaluate_accuracy(ccp_tree, X_test, y_test)
        nodes_ccp = count_nodes(ccp_tree)
        depth_ccp = get_tree_depth(ccp_tree)
        
        # Record results
        unpruned_accuracies.append(acc_unpruned)
        pruned_accuracies.append(acc_pruned)
        ccp_accuracies.append(acc_ccp)
        node_counts_unpruned.append(nodes_unpruned)
        node_counts_pruned.append(nodes_pruned)
        node_counts_ccp.append(nodes_ccp)
        depths_unpruned.append(depth_unpruned)
        depths_pruned.append(depth_pruned)
        depths_ccp.append(depth_ccp)
        
        print(f"Unpruned tree: {nodes_unpruned} nodes, {acc_unpruned:.2%} accuracy")
        print(f"REP tree: {nodes_pruned} nodes, {acc_pruned:.2%} accuracy")
        print(f"CCP tree: {nodes_ccp} nodes, {acc_ccp:.2%} accuracy")
    
    # Summarize results
    print("\n=== Cross-Validation Summary ===")
    print(f"Unpruned tree: {np.mean(node_counts_unpruned):.1f} nodes (avg),{np.mean(depths_unpruned):.1f} depth (avg),  {np.mean(unpruned_accuracies):.2%} accuracy (avg)")
    print(f"REP tree: {np.mean(node_counts_pruned):.1f} nodes (avg),{np.mean(depths_pruned):.1f} depth (avg), {np.mean(pruned_accuracies):.2%} accuracy (avg)")
    print(f"CCP tree: {np.mean(node_counts_ccp):.1f} nodes (avg),{np.mean(depths_ccp):.1f} depth (avg), {np.mean(ccp_accuracies):.2%} accuracy (avg)")
    
    # Find the best method
    avg_accs = [np.mean(unpruned_accuracies), np.mean(pruned_accuracies), np.mean(ccp_accuracies)]
    methods = ["Unpruned", "Reduced Error Pruning", "Cost Complexity Pruning"]
    best_method_index = np.argmax(avg_accs)
    
    print(f"\nBest method: {methods[best_method_index]} with {avg_accs[best_method_index]:.2%} average accuracy")

def plot_accuracy_vs_dataset_size(data, feature_names, max_dataset_size=None, step_size=1000):

    accuracies = []
    dataset_sizes = []

    if max_dataset_size is None:
        max_dataset_size = len(data)

    for size in range(step_size, max_dataset_size + 1, step_size):
        print(f"Training and evaluating on {size} samples...")


        data_subset = data[:size]

        accuracy, _ =k_fold(data_subset, k=5, seed=70)

        print(accuracy)
        accuracies.append(accuracy)
        dataset_sizes.append(size)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, accuracies, marker='o', linestyle='-', color='blue')
    plt.title("Decision Tree Accuracy vs. Dataset Size")
    plt.xlabel("Dataset Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    print("started training")

    dataset, feature_names =read_dt_csv("dt.csv")

    tree=train_tree(dataset, feature_names)
    
    #avg_acc, std_acc=k_fold(dataset, k=10, seed=70)
    #print(f"\nAverage accuracy across {10} folds: {avg_acc:.2%}")
    #print(f"\nAverage std deviation across {10} folds: {std_acc:.2%}")

    #plot_accuracy_vs_dataset_size(dataset, feature_names=feature_names,max_dataset_size=None, step_size=2000) 

    #X = dataset[:, :-1]  
    #y = dataset[:, -1].astype(int)
    
    

    
    save_tree(tree=tree,filename="tree_bad_weights")

    # tree=load_tree(filename="tree_weights.npy")
    
    #compare_pruning_methods(dataset)

    #print("\n\n=== K-Fold Cross-Validation with Pruning ===")
    #k_fold_with_pruning(dataset, k=5)
    

    #hyperparameter_tuning

    #best_depth, best_samples = hyperparameter_tuning(dataset)

    #best max_depth= None, best_min_samples=2

    #train_tree(dataset, feature_names, max_depth=best_depth, min_samples=best_samples, test_ratio=0.2)

    #avg_acc, std_acc=k_fold(dataset, k=10, seed=70,max_depth=best_depth, min_samples=best_samples)
    #print(f"\nAverage accuracy across {10} folds: {avg_acc:.2%}")
    #print(f"\nAverage std deviation across {10} folds: {std_acc:.2%}")


