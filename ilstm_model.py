import torch
import torch.nn as nn


class ILSTM(nn.Module):
    """
    Incremental LSTM with persistent but detached hidden state.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, x):
        # Detach hidden state to avoid backprop through old graphs
        if self.hidden_state is not None:
            h, c = self.hidden_state
            self.hidden_state = (h.detach(), c.detach())

        out, self.hidden_state = self.lstm(x, self.hidden_state)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
