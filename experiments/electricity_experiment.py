import os
import numpy as np

from src.datasets.electricity import load_electricity_data
from src.models.ilstm import build_ilstm
from src.training.incremental_trainer import run_incremental_training


def main():
    # -------------------------
    # Paths
    # -------------------------
    DATA_PATH = os.path.join("data", "electricity.csv")

    # -------------------------
    # Hyperparameters
    # -------------------------
    BATCH_SIZE = 336          # 1 week of data
    SEQUENCE_LENGTH = 48      # 1 day of data
    INITIAL_EPOCHS = 300
    INCREMENTAL_EPOCHS = 60

    # -------------------------
    # Load data
    # -------------------------
    X, y = load_electricity_data(DATA_PATH)
    print(f"Loaded data: X={X.shape}, y={y.shape}")

    # -------------------------
    # Build model
    # -------------------------
    n_features = X.shape[1]
    model = build_ilstm(
    batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    n_features=n_features
)

    model.summary()

    # -------------------------
    # Run incremental training
    # -------------------------
    accuracies = run_incremental_training(
        model=model,
        X=X,
        y=y,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        initial_epochs=INITIAL_EPOCHS,
        incremental_epochs=INCREMENTAL_EPOCHS,
    )

    # -------------------------
    # Results
    # -------------------------
    print("\nIncremental training finished.")


if __name__ == "__main__":
    main()
