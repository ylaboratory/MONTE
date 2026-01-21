<img src="assests/monte.png" alt="MONTE icon" width="150" height="150"/>

# MONTE

**MONTE**: Methylation-based Observation Normalization and Tumor purity Estimation

> [!NOTE]
> **Slow git clone speed:** If you experience slow download speeds, this is because we've included all pretrained models in this repository. In a future official release, we plan to move the pretrained models to cloud storage to improve download performance. Thank you for your patience.

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
    - [Methylation purification](#methylation-purification)
    - [Bayesian transfer learning](#bayesian-transfer-learning)
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
  The beta values should be stored in a **pandas DataFrame** with the shape `(samples, probes)`. It is essential that the **column names** (i.e., probe names) are provided, as they will be used to align probe features when applying the model to a new dataset. If your new dataset contains a different set of probes, the model will automatically compute the intersection of the probe sets.

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

# Assume your beta value table is X, and the y is your purity scores
model.fit(X, y)
```

### Using pretrained models

Another way to use this package is through the **pretrained models**, which allow users to skip the training process. This package provides **22** pretrained models which trained on the 450K methylation microarray from the TCGA.  

To view the available pretrained models, you can use `available_pretrained_models()`. And you can further access the model by `from_pretrained(model_name)`.

```python
from monte import available_pretrained_models, from_pretrained

# print all pretrained models
print(available_pretrained_models())

# access to the cancer-specific model
model = from_pretrained("COAD")

# access to the pan-cancer model
pan_cancer_model = from_pretrained("PAN-CANCER")
```

The `model` works identically to the manually trained models, all the functions are avaiable.

### Purity esitmation

To estimate tumor purity scores, either train your own model or use a pretrained model. Then, pass your input beta values (or M-values) matrix to the `predict_purity` function.

```python
estimated_purity = model.predict_purity(your_data)
```

### Methylation purification

To purifiy the methlyation values, the `purify_values()` function will internally esitmate the purity first and then adjust the values by given target purity you want. In default the target purity is 1.

```python
purified_beta = model.purify_values(your_data, target_purity=1)
```

### Bayesian transfer learning

The MONTE package also provides functionality to leverage existing or pretrained models on new datasets with different purity metrics. Use the `fine_tune` function to adapt the model:

```python
model.fine_tune(your_data, new_purity_scores)
```

This updates the model coefficients and internal statistics to smoothly adapt to your new dataset.

### Save and load

To reuse the model, the package provides functionality to easily save and load MONTE models.

```python
# save model
model.save(your_model_saved_path)

# load model
# since loading a model requires a Monte instance, you can load it using the class method
model = Monte.load(your_model_saved_path)
```

## Analysis

---

## Citation

The preprint will be avaiable soon on bioRxiv.

<!-- If you use MONTE in your research, please cite our preprint:

```bibtex
@article{MONTE2024,
  title={MONTE: Methylation-based Observation Normalization and Tumor purity Estimation},
  author={...},
  journal={bioRxiv},
  year={2024},
  doi={...}
}
```

The preprint is available on bioRxiv and will be updated with the final citation information upon publication. -->
