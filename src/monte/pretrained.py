import os
from monte.main import Monte

# Registry: model name -> file name
_PRETRAINED = {
    "COAD": "COAD.pkl",
    "BRCA": "BRCA.pkl",
    "PAN-CANCER": "PAN-CANCER.pkl",
}


def from_pretrained(name: str) -> Monte:
    if name not in _PRETRAINED:
        raise ValueError(
            f"Unknown pretrained model '{name}'. Available: {list(_PRETRAINED.keys())}"
        )

    path = f"pretrained_models/{name}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained model not found at {path}")

    return Monte.load(path)


def available_pretrained_models() -> list:
    """Returns a list of available pretrained model names."""
    return list(_PRETRAINED.keys())
