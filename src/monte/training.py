import pandas as pd
import numpy as np
from typing import Optional, List
from monte.main import Monte
from sklearn.model_selection import KFold
from copy import deepcopy

def train_with_cv(
    X: pd.DataFrame,
    purity: pd.Series,
    top_n_candidates: Optional[List[int]] = None,
    n_splits: int = 5,
    alpha: float = 0.05,
    eps: float = 1e-10,
) -> "Monte":

    # top_n candidates
    base_candidates = [
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1_000,
        2_500,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
        150_000,
    ]

    if top_n_candidates is not None and len(top_n_candidates) > 0:
        for n in top_n_candidates:
            if isinstance(n, int) and n > 0 and n not in base_candidates:
                base_candidates.append(n)
            else:
                raise ValueError(
                    "top_n_candidates must be a list of positive integers."
                )
    base_candidates = sorted(base_candidates)

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

        model = Monte(alpha=alpha, eps=eps)
        model.fit(X_train, y_train)

        for top_n in top_n_candidates:
            preds = model.predict_purity(X_test, top_n=top_n)
            mse = ((np.asarray(preds) - np.asarray(y_test)) ** 2).mean()
            mse_results[top_n].append(mse)

    mean_mse = {k: np.mean(v) for k, v in mse_results.items()}
    best_top_n = min(mean_mse, key=lambda k: mean_mse[k])

    final_model = Monte(alpha=alpha, eps=eps)
    final_model.fit(X, purity)
    final_model.best_top_n = best_top_n

    return final_model


def fine_tune_with_cv(
    model: "Monte",
    X: pd.DataFrame,
    purity: pd.Series,
    tau2_grids: Optional[List[float]] = None,
    alpha: float = 0.05,
    eps: float = 1e-10,
    n_splits: int = 5,
    top_n: int = 1000,
) -> "Monte":
    
    if not isinstance(model, Monte):
        raise ValueError("model must be an instance of Monte class.")
    
    tau2_grids = tau2_grids
    if tau2_grids is None:
        tau2_grids = [0.01, 0.1, 1.0, 10, 100, 1000]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_results = {tau2: [] for tau2 in tau2_grids}
    for tau2 in tau2_grids:
        for train_idx, test_idx in kf.split(X):
            # Create a deep copy of the model for each fold and tau2 value
            fine_tuned_model = deepcopy(model)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = purity.iloc[train_idx], purity.iloc[test_idx]

            fine_tuned_model = fine_tuned_model.fine_tune(X_train, y_train, tau2=tau2)
            predicted_purity = fine_tuned_model.predict_purity(X_test)
            mse = ((np.asarray(predicted_purity) - np.asarray(y_test)) ** 2).mean()
            mse_results[tau2].append(mse)

    mean_mse = {k: np.mean(v) for k, v in mse_results.items()}
    best_tau2 = min(mean_mse, key=lambda k: mean_mse[k])

    final_model = deepcopy(model)
    final_model = final_model.fine_tune(X, purity, tau2=best_tau2)
    final_model.best_tau2 = best_tau2

    return final_model