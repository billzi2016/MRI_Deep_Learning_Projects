# MRI Deep Learning Projects Overview

This repository contains deep learning projects built on several large public neuroimaging datasets, covering **supervised classification**, **regression**, **self-supervised pretraining**, and **interpretability analysis**.

---

## Datasets

| Folder | Dataset | Summary |
|--------|---------|---------|
| `ABCD/` | Adolescent Brain Cognitive Development | Adolescent brain development cohort with suicide/non-suicide labels and related clinical scale scores |
| `ABIDE_I/` | Autism Brain Imaging Data Exchange I | Autism spectrum disorder vs. healthy control, collected across multiple sites |
| `ABIDE_II/` | Autism Brain Imaging Data Exchange II | The second ABIDE release with a larger sample size and improved acquisition protocols |
| `HCP_Y/` | Human Connectome Project - Young Adult | High-resolution structural and functional imaging for healthy young adults, using the HCP minimal preprocessing pipeline |
| `ADNI/` | Alzheimer's Disease Neuroimaging Initiative | Longitudinal Alzheimer's cohort with five classes: CN / SMC / EMCI / LMCI / AD |

> All datasets are stored in preprocessed `.nii.gz` format, including skull stripping, bias field correction, and registration to MNI152 standard space.

---

## Project Structure

```text
MRI_Deep_Learning_Projects/
├── ABCD_Project/          # ABCD supervised learning
├── ADNI_Project/          # ADNI multi-class classification, regression, and SHAP visualization
├── SimCLR_Project/        # Cross-dataset self-supervised pretraining and feature extraction
├── ABCD/                  # Dataset directory (.nii.gz)
├── ABIDE_I/               # Dataset directory
├── ABIDE_II/              # Dataset directory
├── HCP_Y/                 # Dataset directory
└── ADNI/                  # Dataset directory
```

---

## ABCD_Project: Suicide Risk Classification and Score Regression

Using the ABCD dataset with structural MRI as input, this project trains a **3D VGG16** model for two tasks:

**Model architecture**: 3D VGG16 with 5 convolution blocks, followed by `AdaptiveAvgPool3d -> 4096-dim -> classification/regression head`

| File | Task | Label | Loss |
|------|------|-------|------|
| `train_onehot.py` | Binary classification: suicide vs. non-suicide | `label` (0/1) | `CrossEntropyLoss` |
| `feature_onehot.py` | Extract features for the full dataset | - | - |
| `train_float.py` | Regression: risk score prediction | `score` (MinMax scaled to `[0,1]`) | `MSELoss` |
| `feature_float.py` | Extract features for the full dataset | - | - |

- Classification output: `best_model_onehot.pt`
- Regression output: `best_model_float.pt`
- Feature dimension: **4096-dim**, covering all samples and saved as CSV

---

## ADNI_Project: Alzheimer's Staging and SHAP Interpretability

Using the ADNI dataset, this project performs five-class cognitive staging **classification** and clinical score **regression**, then uses SHAP to locate the brain regions emphasized by the model.

**Five-class label definition** (see `ADNI_Project/README.md` for details):

| Label | Meaning | Core Criteria |
|------|---------|---------------|
| CN | Cognitively normal | CDR = 0, MMSE 24-30, WLM normal |
| SMC | Subjective memory concern | CDR = 0, objective tests normal, self-reported memory decline |
| EMCI | Early mild cognitive impairment | CDR = 0.5, mild WLM impairment |
| LMCI | Late mild cognitive impairment | CDR = 0.5, significant WLM impairment |
| AD | Alzheimer's disease (mild) | CDR = 0.5-1.0, MMSE 20-26 |

| File | Task |
|------|------|
| `train_onehot.py` | Five-class classification with class-weighted `CrossEntropyLoss` |
| `train_float.py` | Clinical score regression with `MSELoss` |
| `shap_vis_onehot.py` | SHAP brain-region visualization for the classification model, including three-view slices and average heatmap NIfTI output |
| `shap_vis_float.py` | SHAP brain-region visualization for the regression model |

- SHAP method: `GradientExplainer`, producing per-subject PNG slices and group-average `.nii.gz` maps

---

## SimCLR_Project: Cross-Dataset Self-Supervised Pretraining

This project uses structural MRI from **all five datasets** as input and applies the **SimCLR** framework for self-supervised contrastive pretraining, learning general-purpose neuroimaging representations before extracting features for downstream tasks.

**Training pipeline:**

```text
Raw MRI -> random augmentation (two views) -> 3D VGG16 encoder -> projector (4096 -> 512 -> 128) -> NT-Xent loss
```

**3D data augmentation strategy:**

- Random flipping along the X / Y / Z axes
- Random Gaussian noise
- Random intensity scaling and shifting
- Random small-angle 3D rotation (`+-10°`)

**NT-Xent loss (normalized temperature-scaled cross entropy):**

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}, \quad \tau = 0.5$$

| File | Function |
|------|----------|
| `train_simclr.py` | Self-supervised pretraining, saving `simclr_encoder.pt` |
| `feature_extract.py` | Load the encoder, extract 4096-dim features for all five datasets, and merge them into `features_all.csv` |

---

## Shared Configuration

| Parameter | Value |
|-----------|-------|
| Input size | `96 x 96 x 96` (resized with trilinear interpolation) |
| Max epochs | 1,000,000 |
| Early stopping patience | 20 |
| GPU | 0-7 (`8 x H100`, `DataParallel`) |
| Mixed precision | AMP (`autocast + GradScaler`) |
| Optimizer | `AdamW` (`weight decay=1e-4`) |

---

## Dependencies

```text
torch >= 2.0
nibabel
pandas
numpy
scipy
scikit-learn
shap
matplotlib
```
