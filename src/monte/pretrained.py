import os
from monte.main import Monte

# Registry: model name -> file name
_PRETRAINED = {
    'HNSC': 'HNSC.pkl',
    'PRAD': 'PRAD.pkl',
    'BLCA': 'BLCA.pkl',
    'BRCA': 'BRCA.pkl',
    'ACC': 'ACC.pkl',
    'LIHC': 'LIHC.pkl',
    'THCA': 'THCA.pkl',
    'SKCM': 'SKCM.pkl',
    'CESC': 'CESC.pkl',
    'OV': 'OV.pkl',
    'COAD': 'COAD.pkl',
    'READ': 'READ.pkl',
    'KIRC': 'KIRC.pkl',
    'KIRP': 'KIRP.pkl',
    'KICH': 'KICH.pkl',
    'UCEC': 'UCEC.pkl',
    'UCS': 'UCS.pkl',
    'LUAD': 'LUAD.pkl',
    'LUSC': 'LUSC.pkl',
    'GBM': 'GBM.pkl',
    'LGG': 'LGG.pkl',
    'PAN-CANCER': 'PAN-CANCER.pkl',
}


def from_pretrained(name: str) -> Monte:
    if name not in _PRETRAINED:
        raise ValueError(
            f"Unknown pretrained model '{name}'. Available: {list(_PRETRAINED.keys())}"
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "pretrained_models", _PRETRAINED[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained model not found at {path}")

    return Monte.load(path)


def available_pretrained_models() -> list:
    """Returns a list of available pretrained model names."""
    return list(_PRETRAINED.keys())
