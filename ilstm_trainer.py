"""
================================================================================
GENERIC INCREMENTAL TRAINING MODULE
================================================================================
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def create_windows(data_x, data_y, window_size):
    """
    Transforms raw data into sequences. 
    The window_size matches the day length (48).
    """
    xs, ys = [], []
    for i in range(len(data_x) - window_size + 1):
        xs.append(data_x[i : i + window_size])
        ys.append(data_y[i + window_size - 1])
    return np.array(xs), np.array(ys)

def run_prequential_experiment(model, stream, config):
    """
    Standard Prequential Loop:
    For each batch: Predict -> Log -> Train -> Repeat.
    """
    history = {"acc": [], "prec": [], "rec": [], "loss": []}
    
    # --- PHASE 1: INITIAL PRE-TRAINING ---
    print(f"\n>>> [1/2] Pre-training on initial {config['initial_batch_size']} samples...")
    init_x, init_y = [], []
    for _ in range(config['initial_batch_size']):
        x, y = next(stream)
        init_x.append(x)
        init_y.append(y)
    
    X_init, y_init = create_windows(init_x, init_y, config['sequence_length'])
    model.fit(X_init, y_init, epochs=config['epochs_initial'], 
              batch_size=config['batch_size'], verbose=0, shuffle=False)

    # --- PHASE 2: STREAMING ---
    print(f"\n>>> [2/2] Streaming Phase Started.")
    print(f"{'Batch':<8} | {'Loss':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 55)

    batch_idx = 1
    try:
        while True:
            # Gather exactly one batch (e.g., 48 samples for 1 day)
            bx, by = [], []
            for _ in range(config['incremental_batch_size']):
                x, y = next(stream)
                bx.append(x)
                by.append(y)
            
            # Format windows
            X_batch, y_batch = create_windows(bx, by, config['sequence_length'])
            
            # 1. TEST: Prequential Evaluation (unseen data)
            loss_val = model.evaluate(X_batch, y_batch, batch_size=config['batch_size'], verbose=0)
            y_prob = model.predict(X_batch, batch_size=config['batch_size'], verbose=0)
            y_pred = (y_prob > 0.5).astype(int)

            # Calculate and Log Metrics
            acc = accuracy_score(y_batch, y_pred)
            prec = precision_score(y_batch, y_pred, zero_division=0)
            rec = recall_score(y_batch, y_pred, zero_division=0)
            
            print(f"{batch_idx:<8} | {loss_val:<8.4f} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f}")

            history["acc"].append(acc)
            history["prec"].append(prec)
            history["rec"].append(rec)
            history["loss"].append(loss_val)
            
            # 2. TRAIN: Incremental Adaptation (60 epochs to avoid 0.000 results)
            model.fit(X_batch, y_batch, epochs=config['epochs_incremental'], 
                      batch_size=config['batch_size'], verbose=0, shuffle=False)
            
            batch_idx += 1
                      
    except StopIteration:
        print("-" * 55)
        print(">>> Stream Finished.")
        
    return history