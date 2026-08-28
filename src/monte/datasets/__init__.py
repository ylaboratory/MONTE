from importlib.resources import files

import pandas as pd


def load_example_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Load the example data and return the train, validation, and test sets."""
    data_dir = files("monte.datasets").joinpath("data")

    with data_dir.joinpath("example_beta_values.csv.gz").open("rb") as file:
        X = pd.read_csv(file, index_col="Barcode", compression="gzip")

    with data_dir.joinpath("example_metadata.csv").open("rb") as file:
        metadata = pd.read_csv(file, index_col=0)

    metadata = metadata.set_index("Barcode")

    train_barcodes = metadata.index[metadata["Set"] == "Train"]
    val_barcodes = metadata.index[metadata["Set"] == "Val"]
    test_barcodes = metadata.index[metadata["Set"] == "Test"]

    X_train = X.loc[train_barcodes]
    X_val = X.loc[val_barcodes]
    X_test = X.loc[test_barcodes]

    y_train = metadata.loc[train_barcodes, "Purity"]
    y_val = metadata.loc[val_barcodes, "Purity"]
    y_test = metadata.loc[test_barcodes, "Purity"]

    return X_train, X_val, X_test, y_train, y_val, y_test


__all__ = ["load_example_data"]