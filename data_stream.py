import pandas as pd
import numpy as np


def electricity_stream(csv_path):
    """
    Sequential data stream for the Electricity dataset.

    This follows the paper's assumption that samples arrive
    one-by-one in time order.
    """
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.int32)

    for i in range(len(X)):
        yield X[i], y[i]
