import pickle
import numpy as np
import pandas as pd
from scipy.stats import t
from itertools import combinations
from typing import List, Tuple, Optional
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm


class Monte:
    def __init__(
        self, alpha: float = 0.05, eps: float = 1e-10
    ):
        self.alpha = alpha
        self.eps = eps
        self.is_fitted: bool = False
        self.is_fine_tuned: bool = False
        self.coef_: pd.Series = pd.Series()  # (m,) tumor direction
        self.intercept_: pd.Series = pd.Series()  # (m,) baseline
        self.w_: pd.Series = pd.Series()  # (m,) probe weights (inverse noise)
        self.probe_ids: List = []
        self.best_top_n: Optional[int] = None
        self.best_tau2: Optional[float] = None

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
        alpha = self.alpha
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
        self, X: pd.DataFrame, target_purity: float = 1.0, alpha: Optional[float] = None
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
        if alpha is None:
            selected_probes = overlap_probes
        else:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha must be in [0, 1]")
            selected_probes = overlap_p_adj[
                overlap_p_adj <= alpha
            ].index.to_list()
            print(
                f"Adjusting {len(selected_probes)} probes passing significance threshold of {alpha}."
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


    def fine_tune(
        self,
        X: pd.DataFrame,
        purity: pd.Series,
        tau2: float = 1,
        top_n: Optional[int] = None
    ) -> "Monte":

        if not self.is_fitted:
            raise ValueError("Base model must be fitted before fine-tuning")
        
        if not isinstance(X, pd.DataFrame) or not isinstance(purity, pd.Series):
            raise ValueError("X must be DataFrame, purity must be Series")

        # --- 1. Select probes ---
        # Find overlapping probes between the model and the new data
        overlap_probes = X.columns.intersection(self.coef_.index)
        if len(overlap_probes) != len(X.columns) or len(overlap_probes) != len(self.coef_.index):
            print(
                f"Warning: Found {len(overlap_probes)} overlapping probes. Fine-tuning will proceed with only these probes."
            )

        # If top_n is specified, select the n-probes with the largest t-statistic from the overlap
        if top_n is not None:
            if top_n > len(overlap_probes):
                print(f"Warning: top_n ({top_n}) is larger than the number of overlapping probes ({len(overlap_probes)}). Using all overlapping probes.")
            selected_probes = self.t_moderated.reindex(overlap_probes).abs().nlargest(top_n).index
        else:
            selected_probes = overlap_probes
        
        if len(selected_probes) == 0:
            raise ValueError("No overlapping probes to fine-tune.")

        # --- 2. Prepare new data ---
        X_arr = X[selected_probes].values.astype(float)
        p_arr = purity.values.astype(float).ravel()
        n_new, m_new = X_arr.shape

        # Center new data
        X_mean_new = X_arr.mean(axis=0)
        p_mean_new = p_arr.mean()
        Xc_new = X_arr - X_mean_new
        pc_new = p_arr - p_mean_new
        pc_new_ss = pc_new @ pc_new
        if pc_new_ss < self.eps:
            raise ValueError("Variance of new purity data is close to zero. Cannot fine-tune.")

        # --- 3. Bayesian Update (vectorized) ---
        # Prior from original fit
        beta_prior = self.coef_.reindex(selected_probes).values
        prior_prec = 1.0 / tau2  # Prior precision

        # Likelihood from new data
        beta_hat_new = (Xc_new.T @ pc_new) / pc_new_ss  # OLS coefficients from new data
        
        # Residual variance from new data
        resid_new = Xc_new - np.outer(pc_new, beta_hat_new)
        dof_new = max(n_new - 2, 1)
        sigma2_new = (resid_new**2).sum(axis=0) / dof_new
        
        # Precision of likelihood from new data
        likelihood_prec = pc_new_ss / (sigma2_new + self.eps)

        # Posterior calculation
        post_prec = prior_prec + likelihood_prec
        post_var = 1.0 / post_prec
        post_mean_beta = post_var * (prior_prec * beta_prior + likelihood_prec * beta_hat_new) # type: ignore

        # --- 4. Update Intercept and Other Stats ---
        post_mean_intercept = X_mean_new - post_mean_beta * p_mean_new
        
        # Credible intervals for the new coefficients
        se_beta = np.sqrt(post_var)
        z_crit = norm.ppf(1 - self.alpha / 2)
        ci_lower = post_mean_beta - z_crit * se_beta
        ci_upper = post_mean_beta + z_crit * se_beta

        # Update df_stats with fine-tuning results
        self.df_stats = pd.DataFrame({
            "coef_prior": self.coef_.reindex(selected_probes),
            "coef_": post_mean_beta,
            "intercept_": post_mean_intercept,
            "lower_CI": ci_lower,
            "upper_CI": ci_upper,
            "posterior_se": se_beta
        }, index=selected_probes)
        
        # --- 5. Store Results ---
        self.coef_ = pd.Series(post_mean_beta, index=selected_probes, name="coef_")
        self.intercept_ = pd.Series(post_mean_intercept, index=selected_probes, name="intercept_")

        # Update other model attributes to reflect the fine-tuned state
        self.probe_ids = list(selected_probes)
        
        # Update means
        self.purity_mean = p_mean_new
        self.probe_mean = pd.Series(X_mean_new, index=selected_probes, name="probe_mean")

        # Update weights and t-statistics
        self.w_ = pd.Series((post_mean_beta**2) / (post_var + self.eps), index=selected_probes, name="weight")
        self.t_moderated = pd.Series(post_mean_beta / (se_beta + self.eps), index=selected_probes, name="t_moderated")

        # Clear stats that are no longer valid
        self.t_raw = pd.Series(dtype='float64')
        self.pvals = pd.Series(dtype='float64')
        self.p_adj = pd.Series(dtype='float64')
        self.is_fine_tuned = True

        return self
    
    def predict_purity_with_ci(
        self,
        X: pd.DataFrame,
        top_n: Optional[int] = None,
        n_simulations: int = 200,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Predicts purity with a confidence interval using a Monte Carlo simulation.
        This method is intended to be used after fine_tuning, as it relies on the
        posterior standard error of the coefficients.
        """
        if "posterior_se" not in self.df_stats.columns:
            raise ValueError(
                "The model has not been fine-tuned. "
                "Run fine_tuning() before calling this method to get confidence intervals."
            )

        if not self.is_fine_tuned:
            raise ValueError("Model has not been fine-tuned. Run fine_tuning() before predicting purity with confidence intervals.")

        selected_probes = self.probe_ids
        if top_n is not None:
            selected_probes = self.t_moderated.abs().nlargest(top_n).index.to_list()
        elif self.best_top_n is not None:
            selected_probes = (
                self.t_moderated.abs().nlargest(self.best_top_n).index.to_list()
            )

        # --- Prepare data and model parameters ---
        X_arr = X.reindex(columns=selected_probes).values.astype(float)
        obs = np.isfinite(X_arr)
        Xc = X_arr - self.probe_mean.reindex(index=selected_probes).values

        coef_mean = self.coef_.reindex(index=selected_probes).values
        coef_se = self.df_stats["posterior_se"].reindex(index=selected_probes).values
        w = self.w_.reindex(index=selected_probes).values

        # --- Monte Carlo Simulation ---
        purity_simulations = np.zeros((X.shape[0], n_simulations))

        for i in range(n_simulations):
            # Sample coefficients from their posterior distribution
            coef_sample = np.random.normal(loc=coef_mean, scale=coef_se) # type: ignore

            # Predict purity with the sampled coefficients
            numerator = np.nansum(obs * w * Xc * coef_sample, axis=1)
            denominator = np.nansum(obs * w * (coef_sample * coef_sample), axis=1) + self.eps
            p_hat_sample = numerator / denominator + self.purity_mean
            purity_simulations[:, i] = p_hat_sample

        # --- Calculate results ---
        p_hat_mean = np.mean(purity_simulations, axis=1)

        alpha = alpha
        lower_bound = np.percentile(purity_simulations, alpha / 2 * 100, axis=1)
        upper_bound = np.percentile(purity_simulations, (1 - alpha / 2) * 100, axis=1)

        results = pd.DataFrame(
            {
                "predicted_purity": np.clip(p_hat_mean, 0.0, 1.0),
                "ci_lower": np.clip(lower_bound, 0.0, 1.0),
                "ci_upper": np.clip(upper_bound, 0.0, 1.0),
            },
            index=X.index,
        )
        return results