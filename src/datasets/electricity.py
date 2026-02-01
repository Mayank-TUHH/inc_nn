import pandas as pd
import numpy as np


def load_electricity_data(csv_path):
    """
    Load and preprocess the Electricity dataset.

    Expected columns:
    feat_1, feat_2, ..., feat_6, target
    """

    df = pd.read_csv(csv_path)

    # Explicit target column
    target_col = "target"

    if target_col not in df.columns:
        raise ValueError("Expected target column 'target' not found.")

    # Split features and labels
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    return X.astype(np.float32), y.astype(np.int32)
