import pickle
import numpy as np
import pandas as pd
from scipy.stats import t
from itertools import combinations
from typing import List, Tuple, Optional
from statsmodels.stats.multitest import multipletests


class Monte:
    def __init__(
        self, significance_level: float = 0.95, eps: float = 1e-10
    ):
        self.significance_level = significance_level
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
    ) -> "Monte":
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
        X_mean = X_arr.mean(axis=0)  # (m,)
        p_mean = p.mean()
        Xc = X_arr - X_mean
        pc = p - p_mean

        # coef_
        denom = pc @ pc
        coef_ = (Xc.T @  pc) / max(denom, self.eps)  # (m,)

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
        vbeta = 1.0 / (pc @ pc)
        t_raw = coef_ / np.sqrt(sig2 * vbeta)
        t_moderated = coef_ / np.sqrt(sig2_post * vbeta)
        df_total = d0 + dof

        # confidence intervals
        alpha = 1 - self.significance_level
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
        self.t_raw = pd.Series(t_raw, index=X.columns, name="t_raw")
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
                "t_raw": t_raw,
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
            selected_probes = (
                self.t_moderated.abs().nlargest(self.best_top_n).index.to_list()
            )

        X_arr = X.reindex(columns=selected_probes).values.astype(float)

        coef = np.asarray(self.coef_.reindex(index=selected_probes))
        w = np.asarray(self.w_.reindex(index=selected_probes))
        obs = np.isfinite(X_arr)

        # center by training mean
        Xc = (
            X_arr - self.probe_mean.reindex(index=selected_probes).values
        )  # store self.X_mean in fit()

        # weighted projection onto coef
        numerator = np.nansum(obs * w * Xc * coef, axis=1)
        denominator = np.nansum(obs * w * (coef * coef), axis=1) + self.eps
        p_hat = (
            numerator / denominator + self.purity_mean
        )  # add back training purity mean

        return pd.Series(
            np.clip(p_hat, 0.0, 1.0), index=X.index, name="predicted_purity"
        )

    def purify_values(
        self, X: pd.DataFrame, target_purity: float = 1.0, significance_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Pure tumor reconstruction from the linear mix: T = intercept_ + (X - intercept_)/p.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame (samples, probes / subset of probes)"
            )
        
        if target_purity < 0.0 or target_purity > 1.0:
            raise ValueError("target_purity must be in [0, 1]")

        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before purifying.")

        # check overlap between input and model probes
        overlap_probes = list(set(X.columns).intersection(set(self.probe_ids)))
        overlap_p_adj = self.p_adj.reindex(index=overlap_probes)
        if len(overlap_probes) == 0:
            raise ValueError(
                "No overlapping probes between input X and model probes."
            )
        elif len(overlap_probes) < len(self.probe_ids):
            print(
                f"Warning: Only {len(overlap_probes)} out of {len(self.probe_ids)} model probes are present in X."
            )
        
        # Determine which probes to use based on significance threshold
        selected_probes = None
        if significance_threshold is None:
            selected_probes = overlap_probes
        else:
            if significance_threshold < 0 or significance_threshold > 1:
                raise ValueError("significance_threshold must be in [0, 1]")
            selected_probes = overlap_p_adj[
                overlap_p_adj <= significance_threshold
            ].index.to_list()
            print(
                f"Adjusting {len(selected_probes)} probes passing significance threshold of {significance_threshold}."
            )
        if len(selected_probes) == 0:
            raise ValueError(
                "No probes pass the significance threshold for purification."
            )

        # prepare data
        X_arr = X.reindex(columns=selected_probes).values.astype(float)

        need_clip = False
        if min(X_arr.flatten()) >= 0.0 and max(X_arr.flatten()) <= 1.0:
            need_clip = True

        # predict purity
        p_pred = np.asarray(self.predict_purity(X))
        delta_p = target_purity - p_pred
        coef_ = np.asarray(self.coef_.reindex(index=selected_probes))
        X_arr += delta_p[:, None] @ coef_[None, :]

        # combine adjusted and unadjusted probes
        unadjusted_probes = list(set(overlap_probes) - set(selected_probes))
        unadjusted_X = X.reindex(columns=unadjusted_probes)
        adjusted_X = pd.DataFrame(X_arr, columns=selected_probes, index=X.index)

        if need_clip:
            adjusted_X = adjusted_X.clip(0.0, 1.0)
        return pd.concat([adjusted_X, unadjusted_X], axis=1).reindex(columns=X.columns)

    def get_probe_stats(self) -> pd.DataFrame:
        """Returns a DataFrame of probe statistics computed during fitting."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Fit before getting stats.")
        return self.df_stats.copy()

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
    def _estimate_prior_params(s2: np.ndarray) -> Tuple:
        from scipy.special import digamma

        """Empirical Bayes hyperparameters (limma-style)."""
        log_s2 = np.log(s2 + 1e-12)
        mean_log = np.mean(log_s2)
        var_log = np.var(log_s2, ddof=1)
        d0 = max(2.0 / var_log, 1.0)
        s0 = np.exp(mean_log - digamma(d0 / 2) + np.log(d0 / 2))
        return s0, d0
