import numpy as np


def run_incremental_training(
    model,
    X,
    y,
    batch_size,
    sequence_length,
    initial_epochs=5,
    incremental_epochs=2,
):
    """
    Run incremental (online) training using prequential evaluation.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled ILSTM model
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    batch_size : int
        Number of samples per batch
    sequence_length : int
        Sequence length for LSTM
    initial_epochs : int
        Epochs for first (pre-training) batch
    incremental_epochs : int
        Epochs for subsequent batches

    Returns
    -------
    accuracies : list
        Accuracy after each batch
    """

    from src.preprocessing.sequence_builder import build_sequences

    n_samples = X.shape[0]
    accuracies = []

    # Split into batches
    batches_X = [
        X[i : i + batch_size] for i in range(0, n_samples, batch_size)
    ]
    batches_y = [
        y[i : i + batch_size] for i in range(0, n_samples, batch_size)
    ]

    # ----- Initial batch (pre-training) -----
    X_init, y_init = build_sequences(
        batches_X[0],
        batches_y[0],
        sequence_length=sequence_length,
    )

    model.fit(
        X_init,
        y_init,
        epochs=initial_epochs,
        batch_size=32,
        verbose=0,
    )

    # ----- Incremental batches -----
    for i in range(1, len(batches_X)):
        X_batch = batches_X[i]
        y_batch = batches_y[i]

        # Build sequences
        X_seq, y_seq = build_sequences(
            X_batch,
            y_batch,
            sequence_length=sequence_length,
        )

        # ----- Prequential evaluation -----
        y_pred = model.predict(X_seq, verbose=0)
        y_pred = (y_pred > 0.5).astype(int).flatten()

        accuracy = np.mean(y_pred == y_seq)
        accuracies.append(accuracy)

        # ----- Incremental training -----
        model.fit(
            X_seq,
            y_seq,
            epochs=incremental_epochs,
            batch_size=32,
            verbose=0,
        )

        print(f"Batch {i} — Accuracy: {accuracy:.4f}")

    return accuracies
