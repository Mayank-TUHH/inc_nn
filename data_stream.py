"""
Electricity data streaming module (paper-aligned).

This file defines how the ELECTRICITY dataset is accessed as a stream,
following the assumptions made in the ILSTM paper.

PAPER-ALIGNED STREAMING PROCEDURE:
----------------------------------
1. The dataset is treated as ONE continuous time-ordered stream.
2. Data is read ONCE from disk.
3. Samples arrive sequentially (simulated streaming).
4. No shuffling, no train/test split.
5. Each sample (x_t, y_t) is seen exactly once.
6. Past samples are never revisited, future samples are never accessed.

This module is responsible ONLY for data arrival.
It does NOT:
- build windows
- batch data
- train models
- compute metrics
"""

import numpy as np
import pandas as pd


def electricity_stream(csv_path):
    """
    Generator that yields electricity data as a time-ordered stream.

    Parameters
    ----------
    csv_path : str
        Path to electricity.csv

    Yields
    ------
    x_t : np.ndarray
        Feature vector at time step t (shape: [n_features])
    y_t : int
        Binary class label at time step t
    """

    # ------------------------------------------------------------
    # Step 1: Load dataset ONCE
    # ------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Step 2: Split features and target
    # ------------------------------------------------------------
    # The electricity dataset is a binary classification problem.
    # The target column is assumed to be named 'target'.
    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.int32)

    # ------------------------------------------------------------
    # Step 3: Sequentially yield samples
    # ------------------------------------------------------------
    # Each (x_t, y_t) pair represents ONE time step in the stream.
    for t in range(len(X)):
        yield X[t], y[t]
