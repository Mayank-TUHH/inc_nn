import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score


def create_sequences(data_x, data_y, seq_length):
    """
    Create overlapping sub-batches using a sliding window.

    This corresponds to Fig. 3 in the paper.
    """
    xs, ys = [], []

    for i in range(len(data_x) - seq_length + 1):
        xs.append(data_x[i : i + seq_length])
        ys.append(data_y[i + seq_length - 1])

    return np.array(xs), np.array(ys)


def run_incremental_training(
    model,
    data_stream,
    sequence_length=48,
    initial_batch_size=960,
    stream_batch_size=336,
    initial_epochs=300,
    stream_epochs=60,
):
    """
    Incremental ILSTM training with prequential evaluation
    """

    loss_fn = BinaryCrossentropy()

    acc_hist, prec_hist, rec_hist, loss_hist = [], [], [], []

    # ------------------------------------------------------------
    # Reset state ONCE before the stream starts
    # ------------------------------------------------------------
    model.reset_states()

    # ------------------------------------------------------------
    # PHASE 1: INITIAL BATCH (960 samples, NO evaluation)
    # ------------------------------------------------------------
    init_x, init_y = [], []

    for _ in range(initial_batch_size):
        x, y = next(data_stream)
        init_x.append(x)
        init_y.append(y)

    init_x = np.array(init_x, dtype=np.float32)
    init_y = np.array(init_y, dtype=np.int32)

    X_init, y_init = create_sequences(init_x, init_y, sequence_length)

    model.fit(
        X_init,
        y_init,
        epochs=initial_epochs,
        shuffle=False,
        batch_size=1,
        verbose=0,
    )

    # ------------------------------------------------------------
    # PHASE 2: STREAMING (336 samples per batch)
    # ------------------------------------------------------------
    while True:
        batch_x, batch_y = [], []

        try:
            for _ in range(stream_batch_size):
                x, y = next(data_stream)
                batch_x.append(x)
                batch_y.append(y)
        except StopIteration:
            break

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.int32)

        X_batch, y_batch = create_sequences(batch_x, batch_y, sequence_length)

        # ---------------- PREQUENTIAL EVALUATION ----------------
        for x_win, y_true in zip(X_batch, y_batch):
            x_win = x_win.reshape(1, sequence_length, -1)

            y_prob = model.predict(x_win, verbose=0)[0, 0]
            y_pred = 1 if y_prob >= 0.5 else 0

            acc_hist.append(accuracy_score([y_true], [y_pred]))
            prec_hist.append(
                precision_score([y_true], [y_pred], zero_division=0)
            )
            rec_hist.append(
                recall_score([y_true], [y_pred], zero_division=0)
            )
            loss_hist.append(loss_fn([y_true], [y_prob]).numpy())

        # ---------------- INCREMENTAL UPDATE ----------------
        model.fit(
            X_batch,
            y_batch,
            epochs=stream_epochs,
            shuffle=False,
            batch_size=1,
            verbose=0,
        )

    return {
        "accuracy": (np.mean(acc_hist), np.std(acc_hist)),
        "precision": (np.mean(prec_hist), np.std(prec_hist)),
        "recall": (np.mean(rec_hist), np.std(rec_hist)),
        "loss": (np.mean(loss_hist), np.std(loss_hist)),
    }
