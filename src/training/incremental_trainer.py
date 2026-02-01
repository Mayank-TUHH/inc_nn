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
    Perform incremental (online) training using a prequential evaluation scheme,
    following the protocol used in the ILSTM reference paper.

    The data stream is split into consecutive batches. The model is first
    pre-trained on an initial batch. For each subsequent batch:

    1. Predictions are generated BEFORE training on the batch.
    2. Performance metrics are computed on the unseen batch.
    3. The model is incrementally updated using the same batch.

    Precision and recall are computed for all batches. In cases where these
    metrics are undefined (e.g., no positive samples or predictions), a value
    of zero is assigned, following the evaluation protocol of the reference
    ILSTM study.

    Aggregate performance is reported as mean ± standard deviation over all
    incremental batches.
    """

    from src.preprocessing.sequence_builder import build_sequences

    bce = BinaryCrossentropy()

    n_samples = X.shape[0]

    acc_list = []
    prec_list = []
    rec_list = []
    loss_list = []

    # --------------------------------------------------
    # Split stream into batches
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
        verbose=1,
    )

    # --------------------------------------------------
    # Incremental batches (prequential evaluation)
    # --------------------------------------------------
    for i in range(1, len(batches_X)):
        X_batch = batches_X[i]
        y_batch = batches_y[i]

        X_seq, y_seq = build_sequences(
            X_batch,
            y_batch,
            sequence_length,
        )

        # -------- PREDICT (before training) --------
        y_prob = model.predict(X_seq, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # -------- METRICS (paper-style) --------
        acc = accuracy_score(y_seq, y_pred)

        prec = precision_score(
            y_seq,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        rec = recall_score(
            y_seq,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        loss = bce(y_seq, y_prob).numpy()

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        loss_list.append(loss)

        # Batch-level logging
        print(
            f"Batch {i:02d} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | "
            f"Loss: {loss:.4f}"
        )

        # -------- TRAIN (after evaluation) --------
        model.fit(
            X_seq,
            y_seq,
            epochs=incremental_epochs,
            batch_size=32,
            verbose=1,
        )

    # --------------------------------------------------
    # Final summary
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

    print("\nIncremental training finished.")

    return acc_list, prec_list, rec_list, loss_list
