import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.losses import BinaryCrossentropy
from src.preprocessing.sequence_builder import build_sequences

def run_incremental_training(
    model, X, y, batch_size, sequence_length, 
    initial_epochs=300, incremental_epochs=60
):
    bce = BinaryCrossentropy()
    
    # 1. Pre-training on the first 960 samples
    X_init_raw, y_init_raw = X[:960], y[:960]
    X_init, y_init = build_sequences(X_init_raw, y_init_raw, sequence_length, batch_size)
    
    print("Pre-training...")
    model.fit(X_init, y_init, epochs=initial_epochs, batch_size=batch_size, shuffle=False, verbose=0)

    # 2. Incremental Training (Weekly batches of 336)
    acc_list = []
    
    # Start after the initial batch
    for i in range(960, len(X) - batch_size, batch_size):
        X_chunk, y_chunk = X[i : i+batch_size+sequence_length], y[i : i+batch_size+sequence_length]
        
        # Prepare sequences for the current weekly batch
        X_seq, y_seq = build_sequences(X_chunk, y_chunk, sequence_length, batch_size)
        
        # PREDICT (Test before Train)
        y_prob = model.predict(X_seq, batch_size=batch_size, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        
        # METRICS
        acc = accuracy_score(y_seq, y_pred)
        acc_list.append(acc)
        
        # TRAIN (Update model with current batch)
        # shuffle=False ensures states represent chronological drift
        model.fit(X_seq, y_seq, epochs=incremental_epochs, batch_size=batch_size, shuffle=False, verbose=0)
        
        print(f"Batch at index {i} - Accuracy: {acc:.4f}")

    print(f"\nMean Accuracy: {np.mean(acc_list)*100:.2f}%")
    return acc_list