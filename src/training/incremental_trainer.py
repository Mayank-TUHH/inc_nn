import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.losses import BinaryCrossentropy


def run_incremental_training(
    model,
    X,
    y,
    batch_size,
    sequence_length,
    initial_epochs=300,
    incremental_epochs=60,
):
    """
    Perform incremental (online) training using a prequential evaluation scheme.

    The data stream is split into consecutive batches. The model is first
    pre-trained on an initial batch. For each subsequent batch, the following
    steps are performed:

    1. The current batch is transformed into overlapping sequences using a
       sliding window approach.
    2. The model generates predictions for the batch (without prior training
       on it).
    3. Performance metrics (accuracy, precision, recall, and loss) are computed
       in a prequential manner.
    4. The model is incrementally updated by training on the same batch for a
       fixed number of epochs.

    This procedure allows the model to adapt continuously to non-stationary
    data without explicit concept drift detection or memory replay.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled LSTM / ILSTM model.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Label vector of shape (n_samples,).
    batch_size : int
        Number of samples per incoming batch.
    sequence_length : int
        Length of the sliding window used to create LSTM input sequences.
    initial_epochs : int
        Number of training epochs for the initial batch.
    incremental_epochs : int
        Number of training epochs for each subsequent batch.

    Returns
    -------
    acc_list : list of float
        Accuracy values for each incremental batch.
    prec_list : list of float
        Precision values for each incremental batch.
    rec_list : list of float
        Recall values for each incremental batch.
    loss_list : list of float
        Binary cross-entropy loss values for each incremental batch.
    """

    from src.preprocessing.sequence_builder import build_sequences

    # Binary cross-entropy for manual loss computation
    bce = BinaryCrossentropy()

    n_samples = X.shape[0]

    acc_list = []
    prec_list = []
    rec_list = []
    loss_list = []

    # --------------------------------------------------
    # Split data stream into batches
    # --------------------------------------------------
    batches_X = [
        X[i : i + batch_size] for i in range(0, n_samples, batch_size)
    ]
    batches_y = [
        y[i : i + batch_size] for i in range(0, n_samples, batch_size)
    ]

    # --------------------------------------------------
    # Initial batch (pre-training)
    # --------------------------------------------------
    X_init, y_init = build_sequences(
        batches_X[0],
        batches_y[0],
        sequence_length,
    )

    model.fit(
        X_init,
        y_init,
        epochs=initial_epochs,
        batch_size=32,
        verbose=0,
    )

    # --------------------------------------------------
    # Incremental batches
    # --------------------------------------------------
    for i in range(1, len(batches_X)):
        X_batch = batches_X[i]
        y_batch = batches_y[i]

        # Build sequences for current batch
        X_seq, y_seq = build_sequences(
            X_batch,
            y_batch,
            sequence_length,
        )

        # Prequential prediction
        y_prob = model.predict(X_seq, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_seq, y_pred)
        prec = precision_score(y_seq, y_pred, zero_division=0)
        rec = recall_score(y_seq, y_pred, zero_division=0)
        loss = bce(y_seq, y_prob).numpy()

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        loss_list.append(loss)

        # Batch-level logging (your requested format)
        print(
            f"Batch {i:02d} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | "
            f"Loss: {loss:.4f}"
        )

        # Incremental training on current batch
        model.fit(
            X_seq,
            y_seq,
            epochs=incremental_epochs,
            batch_size=32,
            verbose=0,
        )

    # --------------------------------------------------
    # Final summary (mean ± std)
    # --------------------------------------------------
    print("\nFinal Performance (mean ± std):")
    print(
        f"Accuracy : {np.mean(acc_list)*100:.2f} ± {np.std(acc_list)*100:.2f}"
    )
    print(
        f"Precision: {np.mean(prec_list)*100:.2f} ± {np.std(prec_list)*100:.2f}"
    )
    print(
        f"Recall   : {np.mean(rec_list)*100:.2f} ± {np.std(rec_list)*100:.2f}"
    )
    print(
        f"Loss     : {np.mean(loss_list):.4f} ± {np.std(loss_list):.4f}"
    )

    return acc_list, prec_list, rec_list, loss_list
