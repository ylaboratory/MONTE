<img src="assests/monte.png" alt="MONTE icon" width="150" height="150"/>

# MONTE

**MONTE**: Methylation-based Observation Normalization and Tumor purity Estimation

---

## Quick naviagation

- [MONTE](#monte)
  - [Quick naviagation](#quick-naviagation)
  - [Installation](#installation)
  - [Instruction and usages](#instruction-and-usages)
    - [Input data](#input-data)
    - [Training the model](#training-the-model)
    - [Using pretrained models](#using-pretrained-models)
    - [Purity esitmation](#purity-esitmation)
    - [Beta adjustment](#beta-adjustment)
    - [Save and load](#save-and-load)
  - [Analysis](#analysis)
  - [Citation](#citation)

## Installation

```python
# download the repo
git clone https://github.com/ylaboratory/MONTE.git
cd MONTE

# you can install it in your python environment via your favorite package manager
conda activate your_env_name
# mamba activate your_env_name

# in the MONTE folder
pip install .
```

## Instruction and usages

### Input data

There are two types of input data required for model training: **beta values** and **purity scores**.

- **Beta Values:**  
  The beta values should be stored in a **pandas DataFrame** with the shape `(samples, probes)`. It is essential that the **column names** (i.e., probe names) are provided, as they will be used to align probe features when applying the model to a new dataset.   If your new dataset contains a different set of probes, the model will automatically compute the intersection of the probe sets.

- **Purity Scores:**
  The purity scores should be provided as a **pandas Series**.

> [!NOTE]
> **Using a Pretrained Model:** If you are using a pretrained model, only the **beta values** are required.

### Training the model

To train your own model, the `Monte` provides a similar interface like the `scikit-learn`, which creating a model object and `fit` / `fit_transform` with you data. The following scripts shows how to train a model, except the input data beta values, and purity

```python
import torch
from monte import Monte

# Create the model
# user can use default values directly
model = Monte()

# Here showing all the arguments with default values
model = Monte(
    max_iter = 200,
    tol = 1e-8,
    device = "cpu",
    dtype = torch.float64,
    random_state = 42,
    verbose = False,
)

# Assume your beta value table is X, and the y is your purity scores
model.fit(X, y, batch_size=None)
```

### Using pretrained models

Another way to use this package is through the **pretrained models**, which allow users to skip the training process. This package provides **??** pretrained models which trained on the 450K methylation microarray from the TCGA.  

To view the available pretrained models, you can use `available_pretrained_models()`. And you can further access the model by `from_pretrained(model_name)`.

```python
from monte import available_pretrained_models, from_pretrained

# print all pretrained models
print(available_pretrained_models())

# access to the pretrained model
model = from_pretrained("COAD")
```

The `model` works identically to the manually trained models, all the functions are avaiable.

### Purity esitmation



### Beta adjustment

### Save and load

## Analysis

---

## Citation
