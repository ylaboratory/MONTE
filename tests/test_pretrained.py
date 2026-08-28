from unittest.mock import patch

import pytest

from monte.pretrained import (
    available_pretrained_models,
    from_pretrained,
)


def test_available_pretrained_models():
    models = available_pretrained_models()

    assert "BRCA" in models
    assert "PAN-CANCER" in models
    assert len(models) == 22


@patch("monte.pretrained.Monte.load")
@patch("monte.pretrained.hf_hub_download")
def test_from_pretrained(mock_download, mock_load, tmp_path):
    mock_download.return_value = "/cached/models/BRCA.pkl"
    expected_model = object()
    mock_load.return_value = expected_model

    model = from_pretrained(
        "brca",
        cache_dir=tmp_path,
    )

    mock_download.assert_called_once_with(
        repo_id="ylab/MONTE-pretrained",
        filename="models/BRCA.pkl",
        revision="v1.0.0",
        cache_dir=tmp_path,
    )
    mock_load.assert_called_once_with("/cached/models/BRCA.pkl")

    assert model is expected_model


@patch("monte.pretrained.Monte.load")
@patch("monte.pretrained.hf_hub_download")
def test_pan_cancer_name(mock_download, mock_load):
    mock_download.return_value = "/cached/models/PAN-CANCER.pkl"

    from_pretrained("pan_cancer")

    mock_download.assert_called_once_with(
        repo_id="ylab/MONTE-pretrained",
        filename="models/PAN-CANCER.pkl",
        revision="v1.0.0",
        cache_dir=None,
    )


def test_unknown_pretrained_model():
    with pytest.raises(ValueError, match="Unknown pretrained model"):
        from_pretrained("UNKNOWN")