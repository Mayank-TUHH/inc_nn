from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def build_ilstm_model(batch_size, seq_length, n_features):
    """
    Build a stateful ILSTM model as described in the paper.
    """
    model = Sequential()
    model.add(
        LSTM(
            150,
            return_sequences=True,
            stateful=True,
            batch_input_shape=(batch_size, seq_length, n_features),
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
