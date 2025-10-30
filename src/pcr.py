import os
import json
import datetime
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

train_path = "/grain/mk98/cancer-methyl/TCGA_Methylation_450K/processed/tumor/train"
test_path = "/grain/mk98/cancer-methyl/TCGA_Methylation_450K/processed/tumor/test"
meta_path = "/grain/mk98/cancer-methyl/TCGA_Methylation_450K/processed/metadata"
out_path = "/grain/mk98/cancer-methyl/pcr_results"

# create timestamped output directory
_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(out_path, _TS)
os.makedirs(out_dir, exist_ok=True)

# configure logging to print timestamps and write to a run log in the out_dir
log_path = os.path.join(out_dir, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
)

def ts_print(msg, level="info"):
    """Timestamped print wrapper that also logs to file."""
    if level == "info":
        logging.info(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)
    else:
        logging.debug(msg)

cancer_meta = pd.read_csv(f"{meta_path}/tumor_metadata.csv").set_index("Barcode")
xy_probes = pd.read_csv(f"/grain/mk98/cancer-methyl/probe_selection_files/xy_probes.txt", header=None)[0].tolist()
purity_col = 'CPE'

def beta_to_m(beta_df, eps=1e-3):
    """Convert β to M-values (clip to avoid infinities)."""
    X = beta_df.clip(eps, 1 - eps)
    return np.log2(X / (1 - X))

def rmse_function(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def choose_pcr_components(M_train, p_train, max_pc=60, cv_splits=5, random_state=0):
    """
    Cross-validate number of PCs (1..max_pc or limited by rank) for PCR.
    Criterion: lowest RMSE (reports tie-broken by higher R^2).
    """
    n = M_train.shape[0]
    rank_cap = min(max_pc, n - 2)
    print(f"Choosing PCR components with rank cap = {rank_cap}")
    pca = PCA(n_components=rank_cap, random_state=random_state)
    PCs = pca.fit_transform(M_train)

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    best_k, best_rmse, best_r2 = 5, float("inf"), -np.inf

    for k in range(2, rank_cap + 1):
        preds, obs = [], []
        for tr, va in kf.split(PCs):
            lr = LinearRegression()
            lr.fit(PCs[tr, :k], p_train.iloc[tr])
            pv = lr.predict(PCs[va, :k])
            preds.append(pv)
            obs.append(p_train.iloc[va].values)
        preds = np.concatenate(preds)
        obs = np.concatenate(obs)
        rmse = rmse_function(obs, preds)
        r2 = r2_score(obs, preds)
        if (rmse < best_rmse - 1e-6) or (np.isclose(rmse, best_rmse) and r2 > best_r2):
            best_k, best_rmse, best_r2 = k, rmse, r2

    return best_k

def pcr_fit_predict(M_train, p_train, M_test, n_pc):
    """Fit PCR with n_pc PCs and predict purity for train/test."""
    pca = PCA(n_components=n_pc, random_state=0)
    PCs_train = pca.fit_transform(M_train)
    PCs_test  = pca.transform(M_test)

    lr = LinearRegression()
    lr.fit(PCs_train, p_train.values)

    p_hat_train = lr.predict(PCs_train)
    p_hat_test  = lr.predict(PCs_test)

    # clip to [0,1]
    p_hat_train = np.clip(p_hat_train, 0, 1)
    p_hat_test  = np.clip(p_hat_test, 0, 1)

    return p_hat_train, p_hat_test, pca, lr

def m_to_beta(M_df):
    """Inverse logit base-2: β = 1 / (1 + 2^{-M})."""
    return 1.0 / (1.0 + np.power(2.0, -M_df))

def per_cpg_ols_params(M_train_df, p_train):
    """
    Fit OLS per CpG: M_ij = alpha_j + gamma_j * (p_i - pbar) + eps.
    Returns alpha (Series), gamma (Series), and pbar (float).
    """
    pbar = float(p_train.mean())
    p_c = (p_train - pbar).values.reshape(-1, 1)         # (n x 1)
    X = np.hstack([np.ones((len(p_train), 1)), p_c])     # (n x 2)
    M_train = M_train_df.values                          # (n x m)

    XtX = X.T @ X
    XtY = X.T @ M_train
    B   = np.linalg.solve(XtX, XtY)                      # (2 x m)

    alpha = pd.Series(B[0, :], index=M_train_df.columns)
    gamma = pd.Series(B[1, :], index=M_train_df.columns)
    return alpha, gamma, pbar

def apply_sample_specific_tumor_only(M_df, p_hat, gamma):
    """
    Sample-specific adjustment: M_tum[i, j] = M[i, j] + gamma[j] * (1 - p_hat[i]).
    M_df: (n x m) M-values for the cohort to adjust (e.g., test set)
    p_hat: length-n vector/Series of predicted purity for those samples
    gamma: length-m Series of per-CpG purity slopes from training
    """
    # broadcast (n x 1) * (1 x m) -> (n x m)
    shift = np.outer((1.0 - np.asarray(p_hat, dtype=float)), gamma.values)
    M_tum = M_df.values + shift
    # ensure columns preserved
    return pd.DataFrame(M_tum, index=M_df.index, columns=M_df.columns)

# for all cancers in the path

cancer_files = os.listdir(train_path)
cancers = [f.split("_")[0] for f in cancer_files if f.endswith("_beta.parquet")]

# accumulator for per-cancer summary metrics
summary_records = []

for cancer in cancers:
    ts_print(f"Processing cancer: {cancer}")

    train_beta = pd.read_parquet(f"{train_path}/{cancer}_beta.parquet")
    train_beta = train_beta.dropna()
    train_beta = train_beta.drop(columns=xy_probes, errors='ignore')

    test_beta = pd.read_parquet(f"{test_path}/{cancer}_beta.parquet")
    test_beta = test_beta.dropna()
    test_beta = test_beta.drop(columns=xy_probes, errors='ignore')

    train_meta = cancer_meta.loc[train_beta.index]
    test_meta = cancer_meta.loc[test_beta.index]

    ts_print(f"train shapes: {train_beta.shape}, {train_meta.shape}")
    ts_print(f"test shapes: {test_beta.shape}, {test_meta.shape}")

    p_train = train_meta[purity_col]
    p_test_true = test_meta[purity_col] if purity_col in test_meta.columns else None

    train_M = beta_to_m(train_beta)
    test_M  = beta_to_m(test_beta)

    n_pc = choose_pcr_components(train_M.values, p_train, max_pc=60, cv_splits=5, random_state=0)
    p_hat_train, p_hat_test, pca_model, lr_head = pcr_fit_predict(train_M.values, p_train, test_M.values, n_pc)

    metrics = {}
    if p_test_true is not None and not p_test_true.isna().all():
        rmse = rmse_function(p_test_true.values, p_hat_test)
        r2   = r2_score(p_test_true.values, p_hat_test)
        pearson_corr = pearsonr(p_test_true.values, p_hat_test)[0]
        metrics.update({"test_rmse": float(rmse), "test_r2": float(r2), "test_pearson": float(pearson_corr)})

    alpha_series, gamma_series, pbar = per_cpg_ols_params(train_M, p_train)

    M_tumor_test = apply_sample_specific_tumor_only(test_M, p_hat_test, gamma_series)

    ts_print(f"[PCR] {cancer}: selected PCs = {n_pc}")
    if metrics:
        ts_print(f"[PCR] test RMSE = {metrics['test_rmse']:.4f}, R2 = {metrics['test_r2']:.3f}, Pearson = {metrics['test_pearson']:.3f}")

    # --- Save artifacts for this cancer into timestamped out_dir ---
    cancer_dir = os.path.join(out_dir, cancer)
    os.makedirs(cancer_dir, exist_ok=True)

    # Save trained models
    try:
        joblib.dump(pca_model, os.path.join(cancer_dir, f"{cancer}_pca.joblib"))
        joblib.dump(lr_head, os.path.join(cancer_dir, f"{cancer}_lr.joblib"))
        ts_print(f"Saved models to {cancer_dir}")
    except Exception as e:
        ts_print(f"Failed to save models for {cancer}: {e}", level="error")

    # Save predictions and metrics
    try:
        pd.Series(p_hat_test, index=test_M.index, name="p_hat_test").to_csv(os.path.join(cancer_dir, f"{cancer}_p_hat_test.csv"))
        pd.Series(p_hat_train, index=train_M.index, name="p_hat_train").to_csv(os.path.join(cancer_dir, f"{cancer}_p_hat_train.csv"))
        with open(os.path.join(cancer_dir, f"{cancer}_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        ts_print(f"Saved predictions and metrics to {cancer_dir}")
    except Exception as e:
        ts_print(f"Failed to save predictions/metrics for {cancer}: {e}", level="error")

    # Save alpha/gamma and adjusted M
    try:
        alpha_series.to_csv(os.path.join(cancer_dir, f"{cancer}_alpha.csv"))
        gamma_series.to_csv(os.path.join(cancer_dir, f"{cancer}_gamma.csv"))
        pd.DataFrame(M_tumor_test, index=M_tumor_test.index, columns=M_tumor_test.columns).to_csv(
            os.path.join(cancer_dir, f"{cancer}_M_tumor_test.csv")
        )
        ts_print(f"Saved alpha/gamma and adjusted M to {cancer_dir}")
    except Exception as e:
        ts_print(f"Failed to save alpha/gamma/adjusted M for {cancer}: {e}", level="error")

    # --- Append summary record for this cancer ---
    try:
        rec = {
            "cancer": cancer,
            "n_train": int(train_M.shape[0]) if train_M is not None else None,
            "n_test": int(test_M.shape[0]) if test_M is not None else None,
            "test_rmse": metrics.get("test_rmse") if metrics else None,
            "test_r2": metrics.get("test_r2") if metrics else None,
            "test_pearson": metrics.get("test_pearson") if metrics else None,
        }
        summary_records.append(rec)
    except Exception as e:
        ts_print(f"Failed to append summary record for {cancer}: {e}", level="error")

# --- End cancer loop ---

# Create a summary DataFrame and save it
try:
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", indent=2)
    ts_print(f"Saved summary for all cancers to {summary_csv} and {summary_json}")
except Exception as e:
    ts_print(f"Failed to save overall summary: {e}", level="error")