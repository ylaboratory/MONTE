from pathlib import Path

from huggingface_hub import hf_hub_download

from monte.main import Monte


_REPO_ID = "ylab/MONTE-pretrained"
_REVISION = "v1.0.0"

_PRETRAINED = {
    "ACC": "models/ACC.pkl",
    "BLCA": "models/BLCA.pkl",
    "BRCA": "models/BRCA.pkl",
    "CESC": "models/CESC.pkl",
    "COAD": "models/COAD.pkl",
    "GBM": "models/GBM.pkl",
    "HNSC": "models/HNSC.pkl",
    "KICH": "models/KICH.pkl",
    "KIRC": "models/KIRC.pkl",
    "KIRP": "models/KIRP.pkl",
    "LGG": "models/LGG.pkl",
    "LIHC": "models/LIHC.pkl",
    "LUAD": "models/LUAD.pkl",
    "LUSC": "models/LUSC.pkl",
    "OV": "models/OV.pkl",
    "PAN-CANCER": "models/PAN-CANCER.pkl",
    "PRAD": "models/PRAD.pkl",
    "READ": "models/READ.pkl",
    "SKCM": "models/SKCM.pkl",
    "THCA": "models/THCA.pkl",
    "UCEC": "models/UCEC.pkl",
    "UCS": "models/UCS.pkl",
}


def available_pretrained_models() -> list[str]:
    """Return the names of the available pretrained models."""
    return list(_PRETRAINED)


def from_pretrained(
    name: str,
    cache_dir: str | Path | None = None,
) -> Monte:
    """Download and load a pretrained MONTE model."""
    name = name.strip().upper().replace("_", "-")

    if name not in _PRETRAINED:
        raise ValueError(
            f"Unknown pretrained model {name!r}. "
            f"Available models: {available_pretrained_models()}"
        )

    model_path = hf_hub_download(
        repo_id=_REPO_ID,
        filename=_PRETRAINED[name],
        revision=_REVISION,
        cache_dir=cache_dir,
    )

    return Monte.load(model_path)