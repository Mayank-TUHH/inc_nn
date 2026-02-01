import numpy as np


def build_sequences(X, y=None, sequence_length=48):
    """
    Convert flat samples into sliding window sequences for LSTM.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray or None
        Labels of shape (n_samples,)
        If None, only X sequences are returned (used during prediction)
    sequence_length : int
        Number of time steps per sequence

    Returns
    -------
    X_seq : np.ndarray
        Shape (n_sequences, sequence_length, n_features)
    y_seq : np.ndarray or None
        Shape (n_sequences,)
    """

    n_samples = X.shape[0]

    if n_samples < sequence_length:
        raise ValueError(
            f"Not enough samples ({n_samples}) for sequence length {sequence_length}"
        )

    X_sequences = []
    y_sequences = []

    for i in range(n_samples - sequence_length + 1):
        X_sequences.append(X[i : i + sequence_length])

        if y is not None:
            # Label is the last time step in the sequence
            y_sequences.append(y[i + sequence_length - 1])

    X_seq = np.array(X_sequences, dtype=np.float32)

    if y is not None:
        y_seq = np.array(y_sequences, dtype=np.int32)
        return X_seq, y_seq

    return X_seq
