from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam


def build_ilstm_model(batch_size, seq_length, n_features):
    """
    Build a stateful ILSTM model (Keras 3 compatible).

    IMPORTANT:
    - batch_shape must be defined via an Input layer
    """

    model = Sequential()

    # Keras 3: batch_shape must be specified here
    model.add(
        Input(
            batch_shape=(batch_size, seq_length, n_features)
        )
    )

    model.add(
        LSTM(
            150,
            return_sequences=True,
            stateful=True,
        )
    )
    model.add(
        LSTM(
            200,
            return_sequences=True,
            stateful=True,
        )
    )
    model.add(
        LSTM(
            50,
            stateful=True,
        )
    )
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
    )

    return model
