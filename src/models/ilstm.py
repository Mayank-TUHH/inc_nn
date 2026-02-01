import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam


def build_ilstm(
    input_shape,
    lstm_units=(150, 200, 50),
    learning_rate=0.001,
):
    model = Sequential()

    # Explicit Input layer (cleaner)
    model.add(Input(shape=input_shape))

    # First LSTM
    model.add(
        LSTM(
            lstm_units[0],
            activation="tanh",
            return_sequences=True,
        )
    )

    # Middle LSTM layers
    for units in lstm_units[1:-1]:
        model.add(
            LSTM(
                units,
                activation="tanh",
                return_sequences=True,
            )
        )

    # Final LSTM
    model.add(
        LSTM(
            lstm_units[-1],
            activation="tanh",
            return_sequences=False,
        )
    )

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
