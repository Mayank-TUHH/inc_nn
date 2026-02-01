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
    Incremental (prequential) training with detailed batch metrics.
    """

    from src.preprocessing.sequence_builder import build_sequences

    bce = BinaryCrossentropy()

    n_samples = X.shape[0]

    acc_list = []
    prec_list = []
    rec_list = []
    loss_list = []

    # Split into batches
    batches_X = [
        X[i : i + batch_size] for i in range(0, n_samples, batch_size)
    ]
    batches_y = [
        y[i : i + batch_size] for i in range(0, n_samples, batch_size)
    ]

    # ---------- Initial batch (pre-training) ----------
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

    # ---------- Incremental batches ----------
    for i in range(1, len(batches_X)):
        X_batch = batches_X[i]
        y_batch = batches_y[i]

        X_seq, y_seq = build_sequences(
            X_batch,
            y_batch,
            sequence_length,
        )

        # Predictions
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

        print(
            f"Batch {i:02d} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | "
            f"Loss: {loss:.4f}"
        )

        # Incremental training
        model.fit(
            X_seq,
            y_seq,
            epochs=incremental_epochs,
            batch_size=32,
            verbose=1,
        )

    # ---------- Final summary ----------
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

    return acc_list, prec_list, rec_list, loss_list
