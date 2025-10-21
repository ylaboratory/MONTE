import os
import torch
from monte.main import Monte

# Registry: model name -> file name
_PRETRAINED = {
    "COAD": "COAD.pt",
    "BRCA": "BRCA.pt",
    "PAN-CANCER": "PAN-CANCER.pt",
}

# Supported score types
_SCORE_TYPES = ["CPE"]


def from_pretrained(name: str, score_type: str) -> Monte:
    if name not in _PRETRAINED:
        raise ValueError(
            f"Unknown pretrained model '{name}'. Available: {list(_PRETRAINED.keys())}"
        )

    if score_type not in _SCORE_TYPES:
        raise ValueError(
            f"Unknown score type '{score_type}'. Available: {_SCORE_TYPES}"
        )

    path = f"pretrained_models/{score_type}/{name}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained H not found at {path}")

    return Monte.load(path)


def available_pretrained_models() -> list:
    """Returns a list of available pretrained model names."""
    return list(_PRETRAINED.keys())
