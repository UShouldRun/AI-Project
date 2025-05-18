import numpy as np
import pandas as pd 
import os

def read_dt_csv():
    str_path = "../../utils"
    file_path = os.path.join(str_path, "dt1.csv")
    df = pd.read_csv(file_path)
    return df

def bitboard_to_board_features(board1, board2):
    board = np.zeros(42, dtype=int)
    for i in range(42):
        mask = 1 << i
        if board1 & mask:
            board[i] = 1
        elif board2 & mask:
            board[i] = 2
        else:
            board[i] = 0
    return board

# Read and transform data
df = read_dt_csv()
X = []
y = []

for index, row in df.iterrows():
    player = row["player"]
    board1 = int(row["board1"])
    board2 = int(row["board2"])
    action = row["action"]
    
    board_features = bitboard_to_board_features(board1, board2)
    
    # Include player as a feature if needed:
    features = np.concatenate(([player], board_features))  # Optional
    # features = board_features  # If you do NOT want to include player
    
    X.append(features)
    y.append(action)

# Create a new DataFrame
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Build column names
column_names = [f"cell_{i}" for i in range(42)]
column_names = ["player"] + column_names  # Include player if used

df_transformed = pd.DataFrame(np.hstack((X, y)), columns=column_names + ["action"])

# Save to CSV
output_path = os.path.join("../../utils", "dt1_transformed.csv")
df_transformed.to_csv(output_path, index=False)


