# ---------------------------------------------------------
# Incremental LSTM (ILSTM) model
# ---------------------------------------------------------

import torch
import torch.nn as nn


class ILSTM(nn.Module):
    """
    Incremental LSTM model with multiple stacked LSTM layers.

    Key characteristics:
    - Uses standard PyTorch LSTM layers (no architectural modification)
    - Preserves hidden and cell states across consecutive batches
    - Stores states independently for each LSTM layer
    - Safely resets states when batch sizes are incompatible

    This implementation follows the approach described in
    Section 3 (Incremental LSTM) of the paper.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Parameters:
            input_size (int): Number of input features
            hidden_sizes (list[int]): Hidden units per LSTM layer
            output_size (int): Number of output classes
        """
        super().__init__()

        # Stacked LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)

        # Final classification layer
        self.fc = nn.Linear(hidden_sizes[2], output_size)

        # Hidden and cell states stored PER layer
        # Each is a tuple: (h_t, c_t)
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

    def forward(self, x):
        """
        Forward pass through the ILSTM.

        Hidden states are reused across batches when compatible
        (same batch size), enabling incremental learning.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, T, F)

        Returns:
            torch.Tensor: Logits of shape (B, C)
        """
        batch_size = x.size(0)

        # ---- LSTM Layer 1 ----
        if self.hidden1 is not None and self.hidden1[0].size(1) == batch_size:
            x, h1 = self.lstm1(x, self.hidden1)
        else:
            x, h1 = self.lstm1(x)

        # ---- LSTM Layer 2 ----
        if self.hidden2 is not None and self.hidden2[0].size(1) == batch_size:
            x, h2 = self.lstm2(x, self.hidden2)
        else:
            x, h2 = self.lstm2(x)

        # ---- LSTM Layer 3 ----
        if self.hidden3 is not None and self.hidden3[0].size(1) == batch_size:
            x, h3 = self.lstm3(x, self.hidden3)
        else:
            x, h3 = self.lstm3(x)

        # Detach states to prevent backpropagation across batches
        self.hidden1 = (h1[0].detach(), h1[1].detach())
        self.hidden2 = (h2[0].detach(), h2[1].detach())
        self.hidden3 = (h3[0].detach(), h3[1].detach())

        # Use last time step for classification
        out = self.fc(x[:, -1, :])
        return out

    def reset_state(self):
        """
        Resets all hidden and cell states.
        Useful for restarting experiments or baselines.
        """
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
