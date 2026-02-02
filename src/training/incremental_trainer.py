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
    
    print("--- Initial Pre-training Phase ---")
    model.fit(X_init, y_init, epochs=initial_epochs, batch_size=batch_size, shuffle=False, verbose=0)
    print("Pre-training complete.\n")

    # 2. Incremental Training (Weekly batches)
    # Metric lists for summary
    acc_list, prec_list, rec_list, loss_list = [], [], [], []
    
    batch_num = 1
    # Start after the initial batch
    for i in range(960, len(X) - (batch_size + sequence_length), batch_size):
        # Extract chunk and build sliding windows
        X_chunk = X[i : i + batch_size + sequence_length]
        y_chunk = y[i : i + batch_size + sequence_length]
        
        X_seq, y_seq = build_sequences(X_chunk, y_chunk, sequence_length, batch_size)
        
        # --- PREDICT (Prequential Evaluation) ---
        y_prob = model.predict(X_seq, batch_size=batch_size, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        
        # --- CALCULATE METRICS ---
        acc = accuracy_score(y_seq, y_pred)
        prec = precision_score(y_seq, y_pred, zero_division=0)
        rec = recall_score(y_seq, y_pred, zero_division=0)
        loss = bce(y_seq, y_prob).numpy()
        
        # Store for final summary
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        loss_list.append(loss)
        
        print(f"Batch {batch_num:03d} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Loss: {loss:.4f}")
        
        # --- TRAIN ---
        model.fit(X_seq, y_seq, epochs=incremental_epochs, batch_size=batch_size, shuffle=False, verbose=0)
        
        batch_num += 1

    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("FINAL INCREMENTAL RESULTS (Mean ± Std)")
    print("="*50)
    print(f"Accuracy:  {np.mean(acc_list)*100:6.2f}% ± {np.std(acc_list)*100:5.2f}%")
    print(f"Precision: {np.mean(prec_list)*100:6.2f}% ± {np.std(prec_list)*100:5.2f}%")
    print(f"Recall:    {np.mean(rec_list)*100:6.2f}% ± {np.std(rec_list)*100:5.2f}%")
    print(f"BCE Loss:  {np.mean(loss_list):6.4f}  ± {np.std(loss_list):5.4f}")
    print("="*50)

    return {"acc": acc_list, "prec": prec_list, "rec": rec_list, "loss": loss_list}