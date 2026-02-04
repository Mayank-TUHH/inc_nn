import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_stream_elec import load_electricity, electricity_stream
from trainer import create_sub_batches, train_ilstm, accuracy, precision_recall
from ilstm_model import ILSTM

csv_path = "data/electricity.csv"

# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------

X, y = load_electricity(csv_path)

stream = electricity_stream(X, y)

X0, y0 = next(stream)
print(X0.shape, y0.shape)

X1, y1 = next(stream)
print(X1.shape, y1.shape)

Xs, ys = create_sub_batches(X1, y1, window_size=48)
print(Xs.shape, ys.shape)

# ---------------------------------------------------------
# Model Hyper Parameters
# ---------------------------------------------------------

# Number of input features and output classes
NUM_FEATURES = 6
NUM_CLASSES = 2

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the Incremental LSTM model
model = ILSTM(
    input_size=NUM_FEATURES,
    hidden_sizes=[150, 200, 50],
    output_size=NUM_CLASSES
).to(device)

# Adam optimizer (same configuration as in the paper)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-7
)

# Loss function for binary/multi-class classification
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------
# Run incremental training with prequential evaluation
# ---------------------------------------------------------

# Create the data stream (generator)
stream = electricity_stream(X, y)

# Train the Incremental LSTM on the Electricity dataset
# Metrics are computed per batch using prequential evaluation
acc_history, prec_history, rec_history = train_ilstm(
    model,
    stream,
    window_size=48
)