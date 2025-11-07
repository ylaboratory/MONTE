import pickle
import numpy as np
import pandas as pd
from scipy.stats import t
from itertools import combinations
from typing import List, Tuple, Optional
from statsmodels.stats.multitest import multipletests

class Monte:
    def __init__(self, lam: float = 1e-6, confidence_level: float = 0.95, eps: float = 1e-10):
        self.lam = lam
        self.confidence_level = confidence_level
        self.eps = eps
        self.is_fitted: bool = False
        self.coef_: pd.Series = pd.Series()  # (m,) tumor direction
        self.intercept_: pd.Series = pd.Series()  # (m,) baseline
        self.w_: pd.Series = pd.Series()  # (m,) probe weights (inverse noise)
        self.probe_ids: List = []
        self.best_top_n: Optional[int] = None
        self.cv_results_: Optional[dict] = None

    # --------------------- FIT ---------------------
    def fit(
        self,
        X: pd.DataFrame,
        purity: pd.Series,
        sample_weights: Optional[np.ndarray] = None,
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
            if not isinstance(sample_weights, np.ndarray):
                raise ValueError("sample_weights must be a np.ndarray (samples,)")
            elif np.isnan(sample_weights).sum() > 0:
                raise ValueError("sample_weights contains NaN values.")
            self.sample_weights_ = sample_weights
        else:
            self.sample_weights_ = np.ones(X.shape[0])
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

        if np.isnan(coef_).sum() > 0:
            raise RuntimeError("NaN values encountered in coefficient estimates.")
        
        #  intercept_
        intercept_ = X_mean - coef_ * p_mean  # (m,)

        # residuals and variances
        fitted = intercept_[None, :] + np.outer(p, coef_)  # (n, m)
        resid = X_arr - fitted  # (n, m)
        dof = max(n - 2, 1)
        sig2 = (resid**2).sum(axis=0) / dof

        # esitmate prior
        s0, d0 = self._estimate_prior_params(sig2)
        self.s0_, self.d0_ = s0, d0

        # posterior variances
        sig2_post = (d0 * s0 + dof * sig2) / (d0 + dof)

        # moderated t-statistics
        vbeta = 1.0 / ((self.sample_weights_ * pc) @ pc + self.lam)
        t_moderated = coef_ / np.sqrt(sig2_post * vbeta)
        df_total = d0 + dof

        # confidence intervals
        alpha = 1 - self.confidence_level
        t_crit = t.ppf(1 - alpha / 2, df_total)
        margin_of_error = t_crit * np.sqrt(sig2_post * vbeta)
        upper_CI = coef_ + margin_of_error
        lower_CI = coef_ - margin_of_error

        # test statistics p-values
        pvals = 2 * (1 - t.cdf(np.abs(t_moderated), df_total))
        _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

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
        self.residual_variance = pd.Series(
            sig2, index=X.columns, name="residual_variance"
        )
        self.moderated_variance = pd.Series(
            sig2_post, index=X.columns, name="moderated_variance"
        )
        self.purity_mean = p_mean
        self.probe_ids = list(X.columns)
        self.is_fitted = True
        self.df_stats = pd.DataFrame(
            {
                "intercept_": intercept_,
                "coef_": coef_,
                "upper_CI": upper_CI,
                "lower_CI": lower_CI,
                "weight": w,
                "t_moderated": t_moderated,
                "p_value": pvals,
                "p_adj": p_adj,
            },
            index=X.columns,
        )

        return self

    def predict_purity(self, X: pd.DataFrame, top_n: Optional[int] = None) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame (samples, probes / subset of probes)"
            )
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before purifying.")

        selected_probes = self.probe_ids
        if top_n is not None:
            selected_probes = self.t_moderated.abs().nlargest(top_n).index.to_list()
        elif self.best_top_n is not None:
            selected_probes = self.t_moderated.abs().nlargest(self.best_top_n).index.to_list()
            print(f"Using best_top_n={self.best_top_n} for prediction.")

        X_arr = X.reindex(columns=selected_probes).values.astype(float)

        coef = np.asarray(self.coef_.reindex(index=selected_probes))
        w = np.asarray(self.w_.reindex(index=selected_probes))
        obs = np.isfinite(X_arr)

        # center by training mean
        Xc = X_arr - self.probe_mean.reindex(index=selected_probes).values  # store self.X_mean in fit()

        # weighted projection onto coef
        numerator = np.nansum(obs * w * Xc * coef, axis=1)
        denominator = np.nansum(obs * w * (coef * coef), axis=1) + self.eps
        p_hat = (
            numerator / denominator + self.purity_mean
        )  # add back training purity mean

        return pd.Series(
            np.clip(p_hat, 0.0, 1.0), index=X.index, name="predicted_purity"
        )
    
    # TODO: add option to select probes used for purification
    def purify_values(self, X: pd.DataFrame) -> pd.DataFrame:
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
        p_pred = np.asarray(self.predict_purity(X))
        delta_p = 1 - p_pred
        coef_ = np.asarray(self.coef_)
        X_arr += delta_p[:, None] @ coef_[None, :]
        return pd.DataFrame(X_arr, index=X.index, columns=self.probe_ids)

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
    def _calc_sample_weights(additional_meta: pd.DataFrame) -> np.ndarray:

        score_cols = ["ABSOLUTE", "ESTIMATE", "LUMP", "IHC"]

        available_cols = [col for col in score_cols if col in additional_meta.columns]
        missing_cols = [col for col in score_cols if col not in additional_meta.columns]
        if missing_cols:
            print(f"Warning: additional_meta missing expected columns: {missing_cols}")

        if len(available_cols) < 2:
            raise ValueError(
                f"Need at least two of {score_cols} to compute agreement; only got {available_cols}"
            )

        scores = additional_meta[available_cols]

        # Parameters controlling penalty for few raters and disagreement
        alpha = 0.5  # penalty for number of available scores
        beta = 1.0   # shape parameter for agreement strength
        min_raters = 2

        def compute_weight(row):
            vals = row.dropna().values
            n = len(vals)
            if n < min_raters:
                return 0.0

            # Mean absolute pairwise difference
            if n == 1:
                mean_abs_diff = 0.0
            else:
                diffs = [abs(a - b) for a, b in combinations(vals, 2)]
                mean_abs_diff = np.mean(diffs)

            # Agreement and weight adjustment
            agreement = 1.0 - mean_abs_diff               # [0,1], higher = more consistent
            rater_factor = (n / len(available_cols)) ** alpha
            w = (agreement ** beta) * rater_factor

            return max(0.0, min(1.0, w))

        weights = scores.apply(compute_weight, axis=1)
        return np.asarray(weights)

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