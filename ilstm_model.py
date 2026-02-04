"""
ILSTM model definition

This file defines the Incremental LSTM (ILSTM) used in the paper.

IMPORTANT PAPER CONSTRAINTS FOLLOWED HERE:
-----------------------------------------
1. This is a STANDARD LSTM architecture (no new gates, no modifications).
2. The model is STATEFUL:
   - hidden state and cell state persist across time
   - states are NOT reset between batches/windows
3. Batch size and sequence length are FIXED.
4. The model itself contains NO streaming logic.
5. The model itself contains NO training loop.
6. The model itself contains NO evaluation logic.

Incrementality comes from HOW the model is trained,
not from how the model is defined.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam


def build_ilstm_model(
    batch_size: int,
    sequence_length: int,
    n_features: int,
):
    """
    Build a stateful ILSTM model for the electricity dataset.

    Parameters
    ----------
    batch_size : int
        Fixed batch size required for stateful LSTM.
        This MUST remain constant during the entire stream.
    sequence_length : int
        Length of the truncated backpropagation window (TBPTT).
    n_features : int
        Number of input features per time step.

    Returns
    -------
    model : keras.Model
        Compiled stateful LSTM model.
    """

    # ------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------
    # A fixed batch_input_shape is REQUIRED for stateful LSTM.
    # This enforces consistent state alignment across time.
    model = Sequential([
        Input(
            batch_shape=(batch_size, sequence_length, n_features)
        ),

        # LSTM layers with persistent hidden & cell states
        LSTM(
            units=150,
            return_sequences=True,
            stateful=True,
        ),

        LSTM(
            units=200,
            return_sequences=True,
            stateful=True,
        ),

        LSTM(
            units=50,
            return_sequences=False,
            stateful=True,
        ),

        # Binary classification output (electricity dataset)
        Dense(
            units=1,
            activation="sigmoid",
        ),
    ])

    # ------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------
    # The optimizer and loss are standard choices.
    # No special incremental optimizer is required by the paper.
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
    )

    return model
