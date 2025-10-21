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
        max_iter: int = 200,
        tol: float = 1e-8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__()
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
    def _init_matrices(self, n_features):
        """Initialize non-negative factors."""
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        def rand(shape):
            return torch.rand(shape, device=self.device, dtype=self.dtype)

        H = rand((2, n_features))
        return H

    # ------------------------------------------------------------------

    def loss_fn(self, H, X, P):
        rec = 0.5 * torch.norm((X - P @ H) ** 2)
        return rec.item()

    def fit(
        self,
        X: pd.DataFrame,
        P: pd.Series,
        batch_size: Optional[int] = None,
    ):
        self.ref_probe_names = X.columns.tolist()

        X_tensor = self._validate_input(X).to(self.device)
        P_tensor = self._validate_purity(P).to(self.device)

        n_samples, n_features = X_tensor.shape
        if batch_size is None:
            batch_size = n_samples
        else:
            batch_size = int(min(batch_size, n_samples))

        # --- Initialize model parameters
        H = self._init_matrices(n_features)

        dataset = TensorDataset(X_tensor, P_tensor)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        history = []
        for it in range(1, self.max_iter + 1):
            # --- iterate over mini-batches
            for Xb, Pb in dataloader:
                numerator = Pb.T @ Xb
                denominator = ((Pb.T @ Pb) @ H).clamp(min=self.eps)
                H *= numerator / denominator
                H = H.clamp(max=1)

            # --- Epoch summary
            with torch.no_grad():
                loss_vals = self.loss_fn(H, X_tensor, P_tensor)
                history.append(loss_vals)

            if self.verbose:
                print(f"Iter {it:03d}: total_loss={loss_vals:.4e}")

        self.H = pd.DataFrame(H.detach().cpu().numpy(), columns=self.ref_probe_names)
        self.history = pd.DataFrame(history, columns=["recon_loss"])
        self.is_fitted = True

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Estimate non-negative W for X ≈ W @ H using NNLS per sample.
        Uses scipy.optimize.nnls (A w = b with A = H.T, b = x).
        """

        self._check_is_fitted()

        probe_names = X.columns.tolist()
        validated_probes = self._validate_probes(self.ref_probe_names, probe_names)

        # convert H to numpy, and select validated probes
        A = self.H[validated_probes].values.T  # (n_features, 2)

        # subset X to validated probes
        X_tensor = self._validate_input(X.loc[:, validated_probes])
        X_np = X_tensor.numpy()  # (n_samples, n_features)

        W_rows = []
        for i in range(X_np.shape[0]):
            b = X_np[i, :]  # (n_features,)
            w, _ = nnls(A, b)  # returns (n_components,)
            W_rows.append(w)

        W_np = np.vstack(W_rows)  # (n_samples, 2)
        W = torch.tensor(W_np, dtype=self.dtype)
        W = W / W.sum(dim=1, keepdim=True)  # normalize rows to sum to 1
        return W.detach().cpu().numpy()

    def fit_transform(
        self, X: pd.DataFrame, P: pd.Series, batch_size: Optional[int] = None
    ) -> np.ndarray:
        self.fit(X, P, batch_size=batch_size)
        W = self.transform(X)
        return W

    def predict_purity(self, X: pd.DataFrame) -> pd.Series:
        self._check_is_fitted()
        W = self.transform(X)
        return pd.Series(W[:, 0])

    def adjust_beta(self, X: pd.DataFrame, target_purity: float = 1) -> pd.DataFrame:
        self._check_is_fitted()

        probe_names = X.columns.tolist()
        validated_probes = self._validate_probes(self.ref_probe_names, probe_names)

        X = X.loc[:, validated_probes]
        H = self.H[validated_probes].to_numpy()

        current_purity = self.transform(X)
        current_purity = torch.tensor(current_purity, dtype=self.dtype)
        X_tensor = self._validate_input(X)
        H_tensor = torch.tensor(H, dtype=self.dtype)

        target_purity_matrix = torch.zeros_like(current_purity)
        target_purity_matrix[:, 0] = target_purity
        target_purity_matrix[:, 1] = 1 - target_purity

        X_adjusted = X_tensor + (target_purity_matrix - current_purity) @ H_tensor

        X_adjusted = X_adjusted.clamp(min=0, max=1)
        return pd.DataFrame(X_adjusted.cpu().numpy(), columns=X.columns, index=X.index)

    def save(self, path: str):
        """Save the learned model parameters and metadata."""
        self._check_is_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "H": self.H.to_dict(),
                "ref_probe_names": self.ref_probe_names,
                "config": {
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
            max_iter=config["max_iter"],
            tol=config["tol"],
            device=config["device"],
            dtype=dtype,
            random_state=config.get("random_state", 42),
            verbose=config["verbose"],
        )

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
            if len(missing_probes) > 0:
                print(
                    f"There are missing probes in the input data. Please check the Monte.get_overlap_probes(your_probes) method for the overlap probes."
                )
            if len(unseen_probes) > 0:
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
