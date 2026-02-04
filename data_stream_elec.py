import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_electricity(csv_path):
    data = pd.read_csv(csv_path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def electricity_stream(X, y, init_batch_size=960, batch_size=336):
    n_samples = X.shape[0]
    start = 0

    # Initial batch
    yield X[start:start + init_batch_size], y[start:start + init_batch_size]
    start += init_batch_size

    # Incremental batches
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]
        start = end


def create_sub_batches(X, y, window_size=48):
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size - 1])

    return np.array(X_seq), np.array(y_seq)
