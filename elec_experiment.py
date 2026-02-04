"""
================================================================================
EXPERIMENT: ELECTRICITY DATASET
================================================================================

This script configures the specific hyperparameters for the Electricity dataset
and triggers the generic training engine.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_stream import electricity_stream
from ilstm_model import build_ilstm_model
from ilstm_trainer import run_prequential_experiment

# ----------------------------------------------------------------------------
# STEP 1: SOURCE OF TRUTH (Hyperparameters)
# ----------------------------------------------------------------------------
# All variables from Section 4.2 of the paper are centralized here.
ELEC_CONFIG = {
    'batch_size': 1,             # Sequential stateful requirement
    'sequence_length': 48,       # 1 day of temporal context
    'n_features': 6,             # Columns in electricity.csv (minus target)
    'initial_batch_size': 960,   # Pre-training size
    'incremental_batch_size': 336, # Adaptation size (1 week)
    'epochs_initial': 300,       # Initial learning iterations
    'epochs_incremental': 60     # Online update iterations
}

# ----------------------------------------------------------------------------
# STEP 2: TOOL INITIALIZATION
# ----------------------------------------------------------------------------
# Build the model and stream using the config above
model = build_ilstm_model(
    ELEC_CONFIG['batch_size'], 
    ELEC_CONFIG['sequence_length'], 
    ELEC_CONFIG['n_features']
)
stream = electricity_stream("data/electricity.csv")

# ----------------------------------------------------------------------------
# STEP 3: EXECUTE & REPORT
# ----------------------------------------------------------------------------
print(f"Starting Incremental Experiment on Electricity Data...")

# Call the generic trainer - it returns the history dictionary
results = run_prequential_experiment(model, stream, ELEC_CONFIG)

# Final Statistical Summary (Mean +/- Std Dev)
print("\n" + "="*50)
print("SCIENTIFIC PERFORMANCE SUMMARY")
print("="*50)
for metric, values in results.items():
    mu = np.mean(values)
    sigma = np.std(values)
    print(f"{metric.upper():<8}: {mu:.4f} \u00B1 {sigma:.4f}")
print("="*50)

# Save visualization of Concept Drift
plt.figure(figsize=(12, 6))
plt.plot(results['acc'], label='Batch Prequential Accuracy', color='skyblue')
# Calculate cumulative average for a smoother trend line
cum_avg = np.cumsum(results['acc']) / (np.arange(len(results['acc'])) + 1)
plt.plot(cum_avg, label='Cumulative Mean Accuracy', color='red', linewidth=2)
plt.title("ILSTM Accuracy Over Evolving Electricity Stream")
plt.xlabel("Batch Index (Weeks)")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("elec_results_analysis.png")