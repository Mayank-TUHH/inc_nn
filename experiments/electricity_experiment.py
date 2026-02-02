import os
import sys
import numpy as np

# Ensure the project root is in path for Colab/CLI imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.electricity import load_electricity_data
from src.models.ilstm import build_ilstm
from src.training.incremental_trainer import run_incremental_training

def main():
    # -------------------------
    # Paths
    # -------------------------
    DATA_PATH = os.path.join(project_root, "data", "electricity.csv")

    # -------------------------
    # Hyperparameter
    # -------------------------
    BATCH_SIZE = 336          # 1 week of data
    SEQUENCE_LENGTH = 48      # 1 day of data (look-back)
    INITIAL_EPOCHS = 300      # Training for the first 960 samples
    INCREMENTAL_EPOCHS = 60   # Training for each subsequent week

    # -------------------------
    # Load data
    # -------------------------
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    X, y = load_electricity_data(DATA_PATH)
    print(f"Loaded Electricity Data: {X.shape[0]} samples, {X.shape[1]} features.")

    # -------------------------
    # Build Model (Stateful)
    # -------------------------
    n_features = X.shape[1]
    
    # We pass batch_size to the builder to enable stateful memory
    model = build_ilstm(
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        n_features=n_features
    )

    model.summary()

    # -------------------------
    # Run Incremental Training
    # -------------------------
    print("\nStarting Incremental Training (Prequential Evaluation)...")
    
    results = run_incremental_training(
        model=model,
        X=X,
        y=y,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        initial_epochs=INITIAL_EPOCHS,
        incremental_epochs=INCREMENTAL_EPOCHS,
    )

    print("\nExperiment finished successfully.")

if __name__ == "__main__":
    main()