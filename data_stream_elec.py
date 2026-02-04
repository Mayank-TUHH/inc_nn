import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_electricity(csv_path):
    """
    Loads and preprocesses the Electricity dataset.

    Steps performed:
    1. Load CSV file from disk
    2. Separate features and target label
    3. Standardize feature values (zero mean, unit variance)

    Standardization is important for LSTM stability and
    faster convergence during training.

    Parameters:
        csv_path (str): Absolute path to electricity.csv

    Returns:
        X (np.ndarray): Feature matrix of shape (N, F)
        y (np.ndarray): Target labels of shape (N,)
    """
    # Load dataset
    df = pd.read_csv(csv_path)

    # Split features and target
    # Assumption: last column is the target label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Normalize features for stable LSTM training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def electricity_stream(
    X,
    y,
    init_batch_size=960,
    batch_size=336
):
    """
    Generator that simulates an online data stream for the Electricity dataset.

    The stream follows the prequential evaluation setup used in the paper:
    - One larger initial batch for pre-training
    - Followed by fixed-size incremental batches

    Parameters:
        X (np.ndarray): Feature matrix of shape (N, F)
        y (np.ndarray): Target labels of shape (N,)
        init_batch_size (int): Size of the initial batch (default: 960)
        batch_size (int): Size of subsequent incremental batches (default: 336)

    Yields:
        (X_batch, y_batch): Tuples containing feature and label batches
    """
    n_samples = len(X)

    # ---- Initial batch (used for pre-training) ----
    yield X[:init_batch_size], y[:init_batch_size]

    # ---- Incremental batches ----
    start = init_batch_size
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]
        start = end


