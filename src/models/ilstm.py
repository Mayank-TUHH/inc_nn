import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_ilstm(
    batch_size,
    sequence_length,
    n_features,
    lstm_units=(150, 200, 50),
    learning_rate=0.001,
):
    model = Sequential()

    # Define the input shape with batch_size explicitly here
    model.add(Input(batch_shape=(batch_size, sequence_length, n_features)))

    model.add(LSTM(
        lstm_units[0],
        activation="tanh",
        return_sequences=True,
        stateful=True
    ))

    # Layer 2
    model.add(LSTM(
        lstm_units[1],
        activation="tanh",
        return_sequences=True,
        stateful=True
    ))

    # Layer 3
    model.add(LSTM(
        lstm_units[2],
        activation="tanh",
        return_sequences=False,
        stateful=True
    ))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model