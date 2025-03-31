import numpy as np
import pandas as pd
import os

from build_dt import build_tree

if __name__ == "__main__":
    
    str_path="../../utils"
    file_path = os.path.join(str_path, "iris.csv")

    df = pd.read_csv(file_path)

    if "ID" in df.columns: 
        df=df.drop(columns=["ID"])

    df["class"]= df["class"].astype("category").cat.codes # Convert class labels to numeric

    iris_data = df.to_numpy()
    X = iris_data[:, :-1]  
    y = iris_data[:, -1].astype(int)
    build_tree(X,y)
    

    
