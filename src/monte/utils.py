import numpy as np
import pandas as pd


def beta_to_m(beta_df: pd.DataFrame, eps=1e-6):
    """Convert β to M-values (clip to avoid infinities)."""
    X = beta_df.clip(eps, 1 - eps)
    return pd.DataFrame(
        np.log2(X / (1 - X)), columns=beta_df.columns, index=beta_df.index
    )
