import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sub_batches(X, y, window_size=48):
    """
    Converts a batch of streaming samples into overlapping sequences
    suitable for LSTM input using a sliding window.

    This implements the sub-batching procedure illustrated in Figure 3
    of the paper.

    Parameters:
        X (np.ndarray): Batch of input features, shape (B, F)
        y (np.ndarray): Batch of labels, shape (B,)
        window_size (int): Number of time steps per sequence

    Returns:
        X_seq (np.ndarray): Array of sequences, shape (B - w + 1, w, F)
        y_seq (np.ndarray): Corresponding labels, shape (B - w + 1,)
                            (label of the last time step in each sequence)
    """
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size - 1])

    return np.array(X_seq), np.array(y_seq)

def train_ilstm(
    model,
    stream,
    window_size=48,
    init_epochs=300,
    inc_epochs=60
):
    """
    Incremental (online) training loop for the ILSTM using
    prequential evaluation.

    For each incoming batch:
      1. Convert the batch into overlapping LSTM sequences
      2. Evaluate the model BEFORE training (prequential evaluation)
      3. Train the model on the same batch
      4. Preserve LSTM states across batches (handled inside the model)

    This follows the experimental protocol described in the paper.

    Parameters:
        model (nn.Module): Incremental LSTM model
        stream (generator): Yields (X_batch, y_batch)
        window_size (int): Sequence length for LSTM input
        init_epochs (int): Training epochs for the first (larger) batch
        inc_epochs (int): Training epochs for subsequent batches

    Returns:
        acc_history  (list[float]): Accuracy per batch
        prec_history (list[float]): Precision per batch
        rec_history  (list[float]): Recall per batch
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Set model to training mode
    model.train()

    # Metric histories (one value per batch)
    acc_history = []
    prec_history = []
    rec_history = []

    # Iterate over the data stream (batch by batch)
    for batch_id, (X_batch, y_batch) in enumerate(stream):

        # ---- Convert batch to LSTM-compatible sequences ----
        X_seq, y_seq = create_sub_batches(X_batch, y_batch, window_size)

        # Move data to PyTorch tensors and the correct device
        X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_seq = torch.tensor(y_seq, dtype=torch.long).to(device)

        # ---- PREQUENTIAL EVALUATION ----
        # Evaluate model BEFORE seeing labels of the current batch
        with torch.no_grad():
            preds = model(X_seq)

            acc = accuracy(preds, y_seq)
            prec, rec = precision_recall(preds, y_seq)

            acc_history.append(acc)
            prec_history.append(prec)
            rec_history.append(rec)

        # ---- TRAINING PHASE ----
        # Use more epochs for the initial batch to bootstrap learning
        epochs = init_epochs if batch_id == 0 else inc_epochs

        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(X_seq)
            loss = criterion(preds, y_seq)
            loss.backward()
            optimizer.step()

        # ---- Logging (for monitoring learning behavior) ----
        print(
            f"Batch {batch_id:02d} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | "
            f"Loss: {loss.item():.4f}"
        )

    return acc_history, prec_history, rec_history

# ---------------------------------------------------------
# Prequential evaluation metric helpers
# ---------------------------------------------------------

def accuracy(preds, labels):
    """
    Computes classification accuracy for a batch.

    Accuracy is defined as the fraction of correctly
    classified samples.

    Parameters:
        preds  (torch.Tensor): Model outputs (logits), shape (N, C)
        labels (torch.Tensor): Ground-truth labels, shape (N,)

    Returns:
        float: Accuracy value in [0, 1]
    """
    return (preds.argmax(dim=1) == labels).float().mean().item()


def precision_recall(preds, labels):
    """
    Computes precision and recall for binary classification.

    Class '1' is treated as the positive class.
    Metrics are computed per batch as part of
    prequential evaluation.

    Parameters:
        preds  (torch.Tensor): Model outputs (logits), shape (N, C)
        labels (torch.Tensor): Ground-truth labels, shape (N,)

    Returns:
        precision (float): Precision value in [0, 1]
        recall    (float): Recall value in [0, 1]
    """
    # Convert logits to predicted class labels
    preds = preds.argmax(dim=1)

    # True positives, false positives, false negatives
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    # Small epsilon avoids division by zero
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    return precision, recall
