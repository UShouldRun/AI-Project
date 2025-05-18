import numpy as np
import pandas as pd
import os

from lib.d_tree.build_dt import build_tree, predict, print_tree 

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

def read_iris_csv():
    str_path="utils"
    file_path = os.path.join(str_path, "iris.csv")

    df = pd.read_csv(file_path)
    
    #preprocess data

    if "ID" in df.columns: 
        df=df.drop(columns=["ID"])

    feature_names = df.columns[:-1].tolist()

    df["class"]= df["class"].astype("category").cat.codes # Convert class labels to numeric
    data = df.to_numpy()

    return data, feature_names

def train_tree(data, feature_names=None, max_depth=None, min_samples=2, test_ratio=0.2):
    X = data[:, :-1]  
    y = data[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=test_ratio)
    decision_tree=build_tree(X_train,y_train)

    print_tree(decision_tree, feature_names)

    predictions = [predict(decision_tree, sample) for sample in X_test]
    
    correct = np.sum(predictions == y_test)  
    accuracy = correct / len(y_test) 

    return decision_tree, accuracy

def hyperparameter_tuning(data):

    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Define parameter grid
    max_depths = [None, 3, 5, 7, 10]
    min_samples = [2, 5, 10]
    
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


if __name__ == "__main__":

    dataset, feature_names =read_iris_csv()

    #train_tree(dataset, feature_names)
    
    print("\nBefore hyperparameter_tuning")
    avg_acc, std_acc=k_fold(dataset, k=10, seed=70)

    print(f"\nAverage accuracy across {10} folds: {avg_acc:.2%}")
    print(f"\nAverage std deviation across {10} folds: {std_acc:.2%}")

    #hyperparameter_tuning

    best_depth, best_samples = hyperparameter_tuning(dataset)

    train_tree(dataset, feature_names, max_depth=None, min_samples=None, test_ratio=0.2)

    avg_acc, std_acc=k_fold(dataset, k=10, seed=70,max_depth=best_depth, min_samples=best_samples)
    print(f"\nAverage accuracy across {10} folds: {avg_acc:.2%}")
    print(f"\nAverage std deviation across {10} folds: {std_acc:.2%}")


