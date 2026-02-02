import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Ensure the project root is in path for Colab/CLI imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.electricity import load_electricity_data
from src.models.ilstm import build_ilstm
from src.training.incremental_trainer import run_incremental_training

def main():
    DATA_PATH = os.path.join(project_root, "data", "electricity.csv")

    # Hyperparameters from Paper
    BATCH_SIZE = 336          
    SEQUENCE_LENGTH = 48      
    INITIAL_EPOCHS = 300      
    INCREMENTAL_EPOCHS = 60   

    # 1. Load Data
    X, y = load_electricity_data(DATA_PATH)
    
    # 2. Normalize Features (Crucial for LSTM Gate Sensitivity)
    # Fit ONLY on the initial training data to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Pre-train portion
    X[:960] = scaler.fit_transform(X[:960])
    # The rest of the stream
    X[960:] = scaler.transform(X[960:])
    
    print("✅ Data normalized to [0, 1] range.")

    # 3. Build Model
    n_features = X.shape[1]
    model = build_ilstm(
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        n_features=n_features
    )

    # 4. Run Incremental Training
    results = run_incremental_training(
        model=model,
        X=X,
        y=y,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        initial_epochs=INITIAL_EPOCHS,
        incremental_epochs=INCREMENTAL_EPOCHS,
    )

    # 5. Final Visualization
    plot_results(results)

def plot_results(results):
    batches = np.arange(1, len(results['acc']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(batches, [a*100 for a in results['acc']], color='green')
    plt.title('Incremental Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(batches, results['loss'], color='red')
    plt.title('BCE Loss')
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()