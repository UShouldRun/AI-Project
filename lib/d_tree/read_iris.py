import numpy as np
import pandas as pd
import os

from build_dt import build_tree, predict, print_tree 

def train_test_split_data(X,y, test_ratio=0.2, seed=70):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size= int(len(X)*test_ratio)
    test_indices= indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
    
def read_csv():
    str_path="../../utils"
    file_path = os.path.join(str_path, "iris.csv")

    df = pd.read_csv(file_path)

    if "ID" in df.columns: 
        df=df.drop(columns=["ID"])

    df["class"]= df["class"].astype("category").cat.codes # Convert class labels to numeric
    iris_data = df.to_numpy()

    return iris_data




if __name__ == "__main__":

    X = iris_data[:, :-1]  
    y = iris_data[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_ratio=0.2)
    decision_tree=build_tree(X_train,y_train)

    print_tree(decision_tree)

    predictions = [predict(decision_tree, sample) for sample in X_test]
    
    print(f"Prediction for sample {X_test}: {predictions}")

    correct = np.sum(predictions == y_test)  
    accuracy = correct / len(y_test) 
    print(f"Accuracy: {accuracy:.2%}")

    
