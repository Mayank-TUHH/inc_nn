import torch
import matplotlib.pyplot as plt

from data_stream_elec import load_electricity, electricity_stream
from ilstm_model import ILSTM
from trainer import train_ilstm_electricity


CSV_PATH = "data/electricity.csv"

NUM_FEATURES = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    X, y = load_electricity(CSV_PATH)
    stream = electricity_stream(X, y)

    model = ILSTM(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)

    acc_history = train_ilstm_electricity(
        model,
        stream,
        device=device
    )

    plt.plot(acc_history)
    plt.xlabel("Stream batch")
    plt.ylabel("Accuracy")
    plt.title("Prequential Accuracy (ILSTM)")
    plt.show()


if __name__ == "__main__":
    main()
