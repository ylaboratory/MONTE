import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Tuple
from scipy.optimize import nnls


class Monte(torch.nn.Module):
    def __init__(
        self,
        n_components: int,
        lam: float = 0.01,
        max_iter: int = 200,
        tol: float = 1e-8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__()
        self.n_components: int = n_components
        self.lam: float = lam
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = dtype
        self.random_state: int = random_state
        self.verbose: bool = verbose
        self.eps: float = torch.finfo(self.dtype).eps

        # parameters created after fitting
        self.ref_probe_names: List[str] = []
        self.H: pd.DataFrame = pd.DataFrame()
        self.history: pd.DataFrame = pd.DataFrame()

        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    def _init_matrices(self, n_samples, n_features):
        """Initialize non-negative factors."""
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        def rand(shape):
            return torch.rand(shape, device=self.device, dtype=self.dtype)

        B = rand((2, n_features))
        W = rand((n_samples, self.n_components))
        H = rand((self.n_components, n_features))
        return B, W, H

    # ------------------------------------------------------------------

    def loss_fn(self, X, P, B, W, H) -> Tuple[float, float]:
        rec = 0.5 * torch.norm(X - P @ B - W @ H) ** 2
        reg = 0.5 * self.lam * torch.norm(B @ H.T) ** 2
        return rec.item(), reg.item()

    def fit(
        self,
        X: pd.DataFrame,
        P: pd.Series,
        block_size: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> "Monte":
        self.ref_probe_names = X.columns.tolist()

        X_tensor = self._validate_input(X).to(self.device)
        P_tensor = self._validate_purity(P).to(self.device)

        n_samples, n_features = X_tensor.shape
        
        # set batch and block sizes
        if batch_size is None:
            batch_size = n_samples
        else:
            batch_size = int(min(batch_size, n_samples))

        if block_size is None:
            block_size = n_features
        else:
            block_size = int(min(block_size, n_features))

        # --- Initialize model parameters
        B, W, H = self._init_matrices(n_samples, n_features)

        indices = torch.arange(n_samples)
        dataset = TensorDataset(X_tensor, P_tensor, indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        history = []

        for it in range(1, self.max_iter + 1):
            for Xb, Pb, idx in dataloader:
                with torch.no_grad():
                    Wb = W[idx]  # batch slice of W

                    # --- Precompute small reusable matrices ---
                    PtP = Pb.T @ Pb                # (2, 2)
                    PtW = Pb.T @ Wb                # (2, n_comp)
                    # HtH = H @ H.T                  # (n_comp, n_comp)
                    # BtB = B @ B.T                  # (2, 2)

                    # === update B blockwise ===
                    for start in range(0, n_features, block_size):
                        end = min(start + block_size, n_features)
                        X_blk = Xb[:, start:end]       # (bs, f_blk)
                        B_blk = B[:, start:end]        # (2, f_blk)
                        H_blk = H[:, start:end]        # (n_comp, f_blk)

                        numerator = Pb.T @ X_blk + 10 * B_blk
                        temp = B_blk @ H_blk.T
                        denominator = (PtP @ B_blk) + (PtW @ H_blk) + self.lam * (temp @ H_blk)
                        B_blk *= numerator / denominator.clamp(min=self.eps)
                        B_blk = B_blk.clamp(min=self.eps, max=1)
                        B[:, start:end] = B_blk

                    # === update W ===
                    BHt = B @ H.T                    # (2, n_comp)
                    HHt = H @ H.T                    # (n_comp, n_comp)
                    numerator = Xb @ H.T             # (bs, n_comp)
                    denominator = Pb @ BHt + Wb @ HHt
                    Wb *= numerator / denominator.clamp(min=self.eps)
                    Wb = Wb.clamp(min=self.eps, max=1)
                    W[idx] = Wb  # write back updates

                    # === update H blockwise ===
                    Wb = W[idx]  # refreshed batch slice of W
                    WtP = Wb.T @ Pb                # (n_comp, 2)
                    WtW = Wb.T @ Wb                # (n_comp, n_comp)

                    for start in range(0, n_features, block_size):
                        end = min(start + block_size, n_features)
                        X_blk = Xb[:, start:end]
                        H_blk = H[:, start:end]
                        B_blk = B[:, start:end]

                        numerator = Wb.T @ X_blk
                        denominator = (WtP @ B_blk) + (WtW @ H_blk) + self.lam * (H_blk @ (B_blk.T @ B_blk))
                        H_blk *= numerator / denominator.clamp(min=self.eps)
                        H_blk = H_blk.clamp(min=self.eps, max=1)
                        H[:, start:end] = H_blk

            # --- Epoch summary ---
            with torch.no_grad():
                loss_vals = self.loss_fn(X_tensor, P_tensor, B, W, H)
                history.append(loss_vals)

                if self.verbose:
                    print(f"Iter {it:03d}: total_loss={loss_vals[0]:.4e}, reg_loss={loss_vals[1]:.4e}")

        self.B = pd.DataFrame(B.detach().cpu().numpy(), columns=self.ref_probe_names)
        self.H = pd.DataFrame(H.detach().cpu().numpy(), columns=self.ref_probe_names)
        self.history = pd.DataFrame(history, columns=["reconstruction_loss", "regularization_loss"])
        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate non-negative W for X ≈ W @ H using NNLS per sample.
        Uses scipy.optimize.nnls (A w = b with A = H.T, b = x).
        """

        self._check_is_fitted()

        probe_names = X.columns.tolist()
        validated_probes = self._validate_probes(self.ref_probe_names, probe_names)

        # convert H to numpy, and select validated probes
        B = self.B[validated_probes].values.T  # (n_features, 2)
        H = self.H[validated_probes].values.T  # (n_features, n_components)
        A = np.concatenate([B, H], axis=1)  # (n_features, 2 + n_components)

        # subset X to validated probes
        X_tensor = self._validate_input(X.loc[:, validated_probes])
        X_np = X_tensor.numpy()  # (n_samples, n_features)

        W_rows = []
        for i in range(X_np.shape[0]):
            b = X_np[i, :]  # (n_features,)
            w, _ = nnls(A, b)  # returns (n_components,)
            W_rows.append(w)

        W_full = np.vstack(W_rows)  # (n_samples, 2 + n_components)
        P_hat = W_full[:, 0:2]  # (n_samples, 2)
        P_hat /= P_hat.sum(axis=1, keepdims=True)
        W_hat = W_full[:, 2:]  # (n_samples, n_components)
        return P_hat, W_hat

    def fit_transform(
        self, X: pd.DataFrame, P: pd.Series, batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(X, P, batch_size=batch_size)
        P_hat, W_hat = self.transform(X)
        return P_hat, W_hat

    def predict_purity(self, X: pd.DataFrame) -> pd.Series:
        self._check_is_fitted()
        P_hat, _ = self.transform(X)
        return pd.Series(P_hat[:, 0])

    def adjust_beta(self, X: pd.DataFrame, target_purity: float = 1) -> pd.DataFrame:
        self._check_is_fitted()

        probe_names = X.columns.tolist()
        validated_probes = self._validate_probes(self.ref_probe_names, probe_names)

        X = X.loc[:, validated_probes]
        B = self.B[validated_probes].to_numpy()

        P_hat, _ = self.transform(X)

        target_purity_matrix = np.zeros_like(P_hat)
        target_purity_matrix[:, 0] = target_purity
        target_purity_matrix[:, 1] = 1 - target_purity

        X_adjusted = (target_purity_matrix - P_hat) @ B

        X_adjusted = X + np.clip(X_adjusted, 0, 1)
        return pd.DataFrame(X_adjusted, columns=X.columns, index=X.index)

    def save(self, path: str):
        """Save the learned model parameters and metadata."""
        self._check_is_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "B": self.B.to_dict(),
                "H": self.H.to_dict(),
                "ref_probe_names": self.ref_probe_names,
                "config": {
                    "n_components": self.n_components,
                    "lam": self.lam,
                    "max_iter": self.max_iter,
                    "tol": self.tol,
                    "dtype": str(self.dtype),
                    "device": str(self.device),
                    "verbose": self.verbose,
                    "random_state": self.random_state,
                },
            },
            path,
        )

        if self.verbose:
            print(f"[Monte] Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load model from saved file."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]

        dtype_str = config["dtype"]
        dtype_attr = dtype_str.split(".")[-1]
        dtype = getattr(torch, dtype_attr, torch.float64)

        model = cls(
            n_components=config["n_components"],
            lam=config["lam"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            device=config["device"],
            dtype=dtype,
            random_state=config.get("random_state", 42),
            verbose=config["verbose"],
        )

        model.B = pd.DataFrame.from_dict(checkpoint["B"])
        model.H = pd.DataFrame.from_dict(checkpoint["H"])
        model.ref_probe_names = checkpoint["ref_probe_names"]
        model.is_fitted = True

        if model.verbose:
            print(f"[Monte] Model loaded from {path}")

        return model

    def get_overlap_probes(self, probes: List[str]) -> List[str]:
        self._check_is_fitted()
        return self._validate_probes(self.ref_probe_names, probes)

    def _validate_probes(self, ref_probes: List[str], probes: List[str]) -> List[str]:
        missing_probes = set(ref_probes) - set(probes)
        unseen_probes = set(probes) - set(ref_probes)

        if len(missing_probes) > 0 or len(unseen_probes) > 0:
            if len(missing_probes) > 0 and self.verbose:
                print(
                    f"There are missing probes in the input data. Please check the Monte.get_overlap_probes(your_probes) method for the overlap probes."
                )
            if len(unseen_probes) > 0 and self.verbose:
                print(
                    f"There are unseen probes in the input data. Please check the Monte.get_overlap_probes(your_probes) method for the overlap probes."
                )

            # Use intersection of probes
            intersection_probes = set(ref_probes) & set(probes)
            if len(intersection_probes) == 0:
                raise ValueError(
                    "No overlapping probes between input data and model reference probes."
                )

            # Warn user about probe mismatch
            print(
                f"Using intersection of available probes for processing. Please check the ordering of output probes carefully."
            )
            return list(intersection_probes)
        return ref_probes

    def _validate_input(self, X: pd.DataFrame) -> torch.Tensor:
        """Validate and convert input X to tensor."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        X_vals = X.values
        if X_vals.ndim != 2:
            raise ValueError("Input X must be a 2D array (n_samples, n_features).")

        if np.any(X_vals < 0) or np.any(X_vals > 1):
            raise ValueError("Input X (beta) values must be in the range [0, 1].")
        return torch.tensor(X_vals, dtype=self.dtype)

    def _validate_purity(self, P: pd.Series) -> torch.Tensor:
        """Validate and convert input purity P to tensor."""
        if not isinstance(P, pd.Series):
            raise ValueError("Input P must be a pandas Series.")

        values = P.values
        # Ensure we have a numeric numpy array for safe scalar comparisons
        values_np = np.asarray(values, dtype=float)
        if values_np.ndim != 1:
            raise ValueError("Input P must be a 1D array (n_samples,).")

        if np.any(values_np < 0) or np.any(values_np > 1):
            raise ValueError("Input P (purity) values must be in the range [0, 1].")

        P_tensor = torch.tensor(values_np, dtype=self.dtype).view(-1, 1)
        return torch.concat([P_tensor, 1 - P_tensor], dim=1)

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling this method.")
