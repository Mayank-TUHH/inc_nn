import numpy as np

def build_sequences(X, y, sequence_length, batch_size):
    """
    Build sliding-window sequences for stateful LSTM.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    sequence_length : int
        Look-back window (e.g., 48 for one day)
    batch_size : int
        Fixed batch size for stateful logic (e.g., 336 for one week)
    """
    X_seq = []
    y_seq = []

    # Sliding window: moves 1 step at a time to cover the whole stream
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i : i + sequence_length])
        y_seq.append(y[i + sequence_length])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int32)

    # Truncate to the nearest multiple of batch_size for Keras stateful mode
    n_batches = len(X_seq) // batch_size
    n_limit = n_batches * batch_size
    
    return X_seq[:n_limit], y_seq[:n_limit]