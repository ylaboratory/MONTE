import pandas as pd
import numpy as np
from typing import Optional
from monte.main import Monte
from sklearn.model_selection import KFold

def train_with_cv(
    X: pd.DataFrame,
    purity: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    n_splits: int = 5,
    lam: float = 1e-6,
    confidence_level: float = 0.95,
    eps: float = 1e-10,
) -> "Monte":
    
    # top_n candidates
    base_candidates = [5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 
                       10_000, 25_000, 50_000, 100_000, 150_000]
    n_probes = X.shape[1]
    max_top_n = min(base_candidates[-1], n_probes)

    top_n_candidates = [n for n in base_candidates if n <= max_top_n]
    if top_n_candidates[-1] < max_top_n:
        top_n_candidates.append(max_top_n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_results = {top_n: [] for top_n in top_n_candidates}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = purity.iloc[train_idx], purity.iloc[test_idx]

        model = Monte(lam=lam, confidence_level=confidence_level, eps=eps)
        model.fit(X_train, y_train, sample_weights=sample_weights)

        for top_n in top_n_candidates:
            preds = model.predict_purity(X_test, top_n=top_n)
            mse = ((preds.values - y_test.values) ** 2).mean()
            mse_results[top_n].append(mse)

    mean_mse = {k: np.mean(v) for k, v in mse_results.items()}
    best_top_n = min(mean_mse, key=lambda k: mean_mse[k])

    final_model = Monte(lam=lam, confidence_level=confidence_level, eps=eps)
    final_model.fit(X, purity, sample_weights=sample_weights)
    final_model.best_top_n = best_top_n
    final_model.cv_results_ = mean_mse

    return final_model
