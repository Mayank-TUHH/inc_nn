"""
================================================================================
GENERIC INCREMENTAL TRAINING MODULE
================================================================================

This module provides a dataset-agnostic framework for incremental learning. 
It follows the "Prequential Evaluation" schema used in data stream mining.

CORE CONCEPTS:
--------------
1. PREQUENTIAL EVALUATION: For every incoming batch, we 'Test' performance on 
   unseen data BEFORE we 'Train' the model on it. This ensures the accuracy 
   metrics are not biased by the model having already seen the labels.
2. STATEFUL ALIGNMENT: Designed for recurrent architectures (LSTM/GRU) where 
   hidden states must persist across time to capture long-term dependencies.
3. SUB-BATCHING: Slices large data arrivals into manageable sliding windows 
   to maintain the temporal context required by recurrent layers.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def create_windows(data_x, data_y, window_size):
    """
    Slices raw data into the sequence format required by LSTM/GRU.
    Each window of size 'w' provides the temporal context for the prediction.
    """
    xs, ys = [], []
    for i in range(len(data_x) - window_size + 1):
        xs.append(data_x[i : i + window_size])
        ys.append(data_y[i + window_size - 1])
    return np.array(xs), np.array(ys)

def run_prequential_experiment(model, stream, config):
    """
    Executes the Prequential (Test-then-Train) loop.
    Prints per-batch metrics and returns full history for summary.
    """
    history = {"acc": [], "prec": [], "rec": [], "loss": []}
    
    # ------------------------------------------------------------------------
    # PHASE 1: INITIAL PRE-TRAINING
    # ------------------------------------------------------------------------
    print(f"\n>>> Starting Phase 1: Pre-training on {config['initial_batch_size']} samples...")
    init_x, init_y = [], []
    for _ in range(config['initial_batch_size']):
        x, y = next(stream)
        init_x.append(x)
        init_y.append(y)
    
    X_init, y_init = create_windows(init_x, init_y, config['sequence_length'])
    
    # Train the base model
    model.fit(X_init, y_init, 
              epochs=config['epochs_initial'], 
              batch_size=config['batch_size'], 
              verbose=0, 
              shuffle=False)
    print(">>> Pre-training complete.\n")

    # ------------------------------------------------------------------------
    # PHASE 2: STREAMING PREQUENTIAL EVALUATION
    # ------------------------------------------------------------------------
    print(f"{'Batch':<8} | {'Loss':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 55)

    batch_idx = 1
    try:
        while True:
            # 1. Gather the incremental batch (e.g., 1 week of data)
            bx, by = [], []
            for _ in range(config['incremental_batch_size']):
                x, y = next(stream)
                bx.append(x)
                by.append(y)
            
            # 2. Convert batch to sequences
            X_batch, y_batch = create_windows(bx, by, config['sequence_length'])
            
            # 3. TEST (Prequential): Evaluate BEFORE training
            loss_val = model.evaluate(X_batch, y_batch, batch_size=config['batch_size'], verbose=0)
            y_prob = model.predict(X_batch, batch_size=config['batch_size'], verbose=0)
            y_pred = (y_prob > 0.5).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_batch, y_pred)
            prec = precision_score(y_batch, y_pred, zero_division=0)
            rec = recall_score(y_batch, y_pred, zero_division=0)

            # Print metrics for the current batch
            print(f"{batch_idx:<8} | {loss_val:<8.4f} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f}")

            # Store metrics
            history["acc"].append(acc)
            history["prec"].append(prec)
            history["rec"].append(rec)
            history["loss"].append(loss_val)
            
            # 4. TRAIN: Adapt the model weights to the new batch
            model.fit(X_batch, y_batch, 
                      epochs=config['epochs_incremental'], 
                      batch_size=config['batch_size'], 
                      verbose=0, 
                      shuffle=False)
            
            batch_idx += 1
                      
    except StopIteration:
        print("-" * 55)
        print(">>> Data stream exhausted.")
        
    return history