import pickle
import numpy as np
import pandas as pd
from typing import List

class Monte:

    def __init__(self, lam: float=1e-6, eps:float =1e-8):
        self.lam = lam
        self.eps = eps
        self.is_fitted:bool    = False
        self.alpha_: pd.Series = pd.Series()  # (m,) baseline
        self.beta_: pd.Series  = pd.Series()  # (m,) tumor direction
        self.w_: pd.Series     = pd.Series()  # (m,) probe weights (inverse noise)
        self.probe_ids: List       = []

    # --------------------- FIT ---------------------
    def fit(self, X: pd.DataFrame, purity: pd.Series) -> "Monte":
        """
        X   : n x m methylation (rows=samples, cols=probes)
        purity: length-n purities in [0,1]
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame (samples x probes)")
        if not isinstance(purity, pd.Series):
            raise ValueError("purity must be a pandas Series (samples,)")
        
        X_arr = X.values.astype(float)
        p = np.asarray(purity, float).ravel()
        n, _ = X_arr.shape

        # Center
        X_mean = X_arr.mean(axis=0)                     # (m,)
        p_mean = p.mean()
        Xc = X_arr - X_mean
        pc = p - p_mean

        # Closed-form ridge for beta (one vector for all probes)
        denom = (pc @ pc) + self.lam               # scalar
        beta = (Xc.T @ pc) / max(denom, self.eps)  # (m,)

        # Alpha from intercept identity
        alpha = X_mean - beta * p_mean             # (m,)

        # Probe noise -> weights
        resid = X_arr - (alpha[None, :] + np.outer(p, beta))   # (n, m)
        # unbiased variance with df≈n-2 (fit alpha & beta)
        dof = max(n - 2, 1)
        sig2 = (resid**2).sum(axis=0) / dof
        w = 1.0 / (sig2 + 1e-6)                    # stable inverse-variance weights

        # Store
        self.alpha_ = pd.Series(alpha, index=X.columns, name="alpha")
        self.beta_  = pd.Series(beta,  index=X.columns, name="beta")
        self.w_     = pd.Series(w,     index=X.columns, name="weight")
        self.probe_ids  = list(X.columns)
        self.is_fitted = True
        return self

    def predict_purity(self, X: pd.DataFrame) -> pd.Series:

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame (samples, probes / subset of probes)")
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before purifying.")

        X_arr = X.reindex(columns=self.probe_ids).values.astype(float)
        a = np.asarray(self.alpha_, dtype=float)
        b = np.asarray(self.beta_, dtype=float)
        w = np.asarray(self.w_, dtype=float)

        # mask of observed entries
        obs = np.isfinite(X_arr)
        # center (leave NaNs)
        Xc = X_arr - a

        # numerator and denominator per row use only observed probes
        num = np.nansum((Xc * b) * w * obs, axis=1)     # (n,)
        den = np.nansum((b**2) * w * obs, axis=1) + self.eps
        p_hat = num / den
        return pd.Series(np.clip(p_hat, 0.0, 1.0), index=X.index, name="predicted_purity")

    def purify_values(self, X: pd.DataFrame, pmin=0.05) -> pd.DataFrame:
        """
        Pure tumor reconstruction from the linear mix: T = alpha + (X - alpha)/p.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame (samples, probes / subset of probes)")
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before purifying.")
        X_arr = X.reindex(columns=self.probe_ids).values.astype(float)

        # predict purity
        p_pred = self.predict_purity(X)

        a = np.asarray(self.alpha_, dtype=float)
        p = np.maximum(np.asarray(p_pred, float).ravel(), pmin)
        T = a + (X_arr - a) / p[:, None]
        return pd.DataFrame(T, index=X.index, columns=self.probe_ids)

    def save(self, filepath: str):
        """Save the entire model into a single binary file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "Monte":
        """Load a previously saved model."""
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Monte):
            raise TypeError("Loaded object is not a Monte instance.")
        return obj