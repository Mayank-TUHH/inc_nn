import torch
import torch.nn as nn
import numpy as np

from data_stream_elec import create_sub_batches


def accuracy(preds, labels):
    return (preds == labels).mean()


def train_ilstm_electricity(
    model,
    stream,
    window_size=48,
    init_epochs=300,
    inc_epochs=60,
    device="cpu"
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    acc_history = []

    for batch_idx, (X_batch, y_batch) in enumerate(stream):

        X_seq, y_seq = create_sub_batches(X_batch, y_batch, window_size)

        X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_seq = torch.tensor(y_seq, dtype=torch.float32).to(device)

        epochs = init_epochs if batch_idx == 0 else inc_epochs

        model.train()
        model.reset_state()

        for _ in range(epochs):
            optimizer.zero_grad()

            outputs = model(X_seq).squeeze(1)   # [N]
            loss = criterion(outputs, y_seq)

            loss.backward()
            optimizer.step()

        # Evaluation on last epoch outputs
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        acc = accuracy(preds, y_seq.cpu().numpy())
        acc_history.append(acc)
        print(f"Batch {batch_idx}, samples: {len(X_batch)}, acc: {acc:.4f}")

    return acc_history
