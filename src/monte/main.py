import pickle
import numpy as np
import pandas as pd
from scipy.stats import t
from typing import List, Tuple, Optional
from statsmodels.stats.multitest import multipletests


class Monte:
    def __init__(self, lam: float = 1e-6, eps: float = 1e-8):
        self.lam = lam
        self.eps = eps
        self.is_fitted: bool = False
        self.coef_: pd.Series = pd.Series()  # (m,) tumor direction
        self.intercept_: pd.Series = pd.Series()  # (m,) baseline
        self.w_: pd.Series = pd.Series()  # (m,) probe weights (inverse noise)
        self.probe_ids: List = []

    # --------------------- FIT ---------------------
    def fit(
        self,
        X: pd.DataFrame,
        purity: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> "Monte":
        """
        X   : n x m methylation (rows=samples, cols=probes)
        purity: length-n purities in [0,1]
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame (samples x probes)")
        if not isinstance(purity, pd.Series):
            raise ValueError("purity must be a pandas Series (samples,)")

        if sample_weights is not None:
            if not isinstance(sample_weights, pd.Series):
                raise ValueError("sample_weights must be a pandas Series (samples,)")
            elif sample_weights.isna().any():
                raise ValueError("sample_weights contains NaN values.")
            self.sample_weights_ = sample_weights.reindex(index=X.index)
        else:
            self.sample_weights_ = pd.Series(
                np.ones(X.shape[0]), index=X.index, name="sample_weights"
            )

        X_arr = X.values.astype(float)
        p = np.asarray(purity, float).ravel()
        n, _ = X_arr.shape

        # Center
        X_mean = X_arr.mean(axis=0)  # (m,)
        p_mean = p.mean()
        Xc = X_arr - X_mean
        pc = p - p_mean

        # coef_
        denom = (self.sample_weights_ * pc) @ pc + self.lam  # scalar
        coef_ = (Xc.T @ (self.sample_weights_ * pc)) / max(denom, self.eps)  # (m,)

        #  intercept_
        intercept_ = X_mean - coef_ * p_mean  # (m,)

        # residuals and variances
        fitted = intercept_[None, :] + np.outer(p, coef_)  # (n, m)
        resid = X_arr - fitted  # (n, m)
        dof = max(n - 2, 1)
        sig2 = (resid**2).sum(axis=0) / dof
        self.residual_variance = pd.Series(
            sig2, index=X.columns, name="residual_variance"
        )

        # esitmate prior
        s0, d0 = self._estimate_prior_params(sig2)
        self.s0_, self.d0_ = s0, d0

        # posterior variances
        sig2_post = (d0 * s0 + dof * sig2) / (d0 + dof)
        self.moderated_variance = pd.Series(
            sig2_post, index=X.columns, name="moderated_variance"
        )

        # moderated t-statistics
        vbeta = 1.0 / ((self.sample_weights_ * pc) @ pc + self.lam)
        t_moderated = coef_ / np.sqrt(sig2_post * vbeta)
        df_total = d0 + dof

        # test statistics p-values
        pvals = 2 * (1 - t.cdf(np.abs(t_moderated), df_total))
        _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        # w = 1.0 / (sig2_used + 1e-6)                    # stable inverse-variance weights
        w = (coef_**2) / (sig2_post + 1e-6)  # SNR weights

        # Store
        self.intercept_ = pd.Series(intercept_, index=X.columns, name="intercept_")
        self.coef_ = pd.Series(coef_, index=X.columns, name="coef_")
        self.w_ = pd.Series(w, index=X.columns, name="weight")
        self.t_moderated = pd.Series(t_moderated, index=X.columns, name="t_moderated")
        self.df_total = df_total
        self.pvals = pd.Series(pvals, index=X.columns, name="p_value")
        self.p_adj = pd.Series(p_adj, index=X.columns, name="p_adj")
        self.probe_mean = pd.Series(X_mean, index=X.columns, name="probe_mean")
        self.purity_mean = p_mean
        self.probe_ids = list(X.columns)
        self.is_fitted = True
        self.df_stats = pd.DataFrame(
            {
                "intercept_": intercept_,
                "coef_": coef_,
                "weight": w,
                "t_moderated": t_moderated,
                "p_value": pvals,
                "p_adj": p_adj,
            },
            index=X.columns,
        )

        return self

    def predict_purity(self, X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame (samples, probes / subset of probes)"
            )
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before purifying.")

        X_arr = X.reindex(columns=self.probe_ids).values.astype(float)
    
        coef = np.asarray(self.coef_)
        w = np.asarray(self.w_)
        obs = np.isfinite(X_arr)
        
        # center by training mean
        Xc = X_arr - self.probe_mean.values  # store self.X_mean in fit()
        
        # weighted projection onto coef
        numerator   = np.nansum(obs * w * Xc * coef, axis=1)
        denominator = np.nansum(obs * w * (coef * coef), axis=1) + self.eps
        p_hat = numerator / denominator + self.purity_mean  # add back training purity mean

        return pd.Series(np.clip(p_hat, 0.0, 1.0), index=X.index, name="predicted_purity")

    def purify_values(self, X: pd.DataFrame, pmin=0.05) -> pd.DataFrame:
        """
        Pure tumor reconstruction from the linear mix: T = intercept_ + (X - intercept_)/p.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame (samples, probes / subset of probes)"
            )
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before purifying.")
        X_arr = X.reindex(columns=self.probe_ids).values.astype(float)

        # predict purity
        p_pred = self.predict_purity(X)

        a = np.asarray(self.intercept_, dtype=float)
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

    @staticmethod
    def _calc_sample_weights(additional_meta: pd.DataFrame) -> pd.Series:
        # check and grab columns from additional meta : ABSOLUTE, ESTIMATE, LUMP, IHC
        score_cols = ["ABSOLUTE", "ESTIMATE", "LUMP", "IHC", "CPE"]
        for col in score_cols:
            if col not in additional_meta.columns:
                print(f"Warning: additional_meta missing expected column '{col}'")
        compare_cols = ["ABSOLUTE", "ESTIMATE", "LUMP", "IHC"]
        scores_cols = additional_meta[score_cols]

        # use elementwise subtraction then abs() to avoid type-checker issues with Series methods
        diffs = [(scores_cols[col] - scores_cols["CPE"]).abs() for col in compare_cols]
        diffs = pd.DataFrame(diffs).T
        score_agreement = diffs.mean(axis=1, skipna=True).values

        weights = np.ones_like(score_agreement, dtype=float) - np.asarray(
            score_agreement, dtype=float
        )
        return pd.Series(weights, index=additional_meta.index, name="sample_weights")

    @staticmethod
    def _estimate_prior_params(s2: np.ndarray) -> Tuple:
        from scipy.special import digamma

        """Empirical Bayes hyperparameters (limma-style)."""
        log_s2 = np.log(s2 + 1e-12)
        mean_log = np.mean(log_s2)
        var_log = np.var(log_s2, ddof=1)
        d0 = max(2.0 / var_log, 1.0)
        s0 = np.exp(mean_log - digamma(d0 / 2) + np.log(d0 / 2))
        return s0, d0
