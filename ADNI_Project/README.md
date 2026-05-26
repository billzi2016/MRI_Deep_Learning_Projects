# ADNI Project — Alzheimer's Disease Stage Classification

## Dataset

**ADNI (Alzheimer's Disease Neuroimaging Initiative)**
Structural MRI (T1-weighted, `.nii.gz`) with clinical stage labels derived from CDR, MMSE, and WLM scores per the ADNI-GO / ADNI-2 / ADNI-3 protocol.

---

## Subject Cohort Definitions

### CN — Cognitively Normal
Healthy control group with fully intact cognitive function.
- CDR global score = 0 (Memory Box must also = 0)
- MMSE: 24–30
- WLM: within education-adjusted normal range

### SMC — Significant Memory Concern *(introduced in ADNI-2)*
Captures the Subjective Cognitive Decline (SCD) population.
- CDR = 0, MMSE: 24–30
- Objective cognitive test performance **completely normal**
- Self- or informant-reported memory decline confirmed via Cognitive Change Index (CCI)

### EMCI — Early Mild Cognitive Impairment
Subtle objective memory impairment, early-stage MCI subgroup.
- CDR = 0.5 (Memory Box ≥ 0.5)
- MMSE: 24–30
- WLM: **mildly impaired** (cutoff between normal and LMCI)

### LMCI — Late Mild Cognitive Impairment
More pronounced memory deficit; roughly equivalent to the original ADNI-1 MCI cohort.
- CDR = 0.5 (Memory Box ≥ 0.5)
- MMSE: 24–30
- WLM: **significantly impaired** (below EMCI threshold)

### AD — Alzheimer's Disease (Dementia Stage)
Meets NINCDS-ADRDA criteria for probable AD. ADNI recruits **mild** AD only (cooperative follow-up requirement).
- CDR = 0.5 or 1.0
- MMSE: 20–26

---

## Label Encoding

| Stage | Integer Label |
|-------|--------------|
| CN    | 0            |
| SMC   | 1            |
| EMCI  | 2            |
| LMCI  | 3            |
| AD    | 4            |

---

## Pipeline

| Script | Description |
|--------|-------------|
| `train_onehot.py` | 3D VGG16 five-class classifier; outputs `best_model_adni.pt` |
| `shap_vis.py`     | SHAP GradientExplainer; saves per-class saliency brain slices |

---

## Notes on Longitudinal Derived Labels
For conversion-prediction tasks (not covered here), MCI subjects are commonly re-labelled as **sMCI** (stable) or **pMCI** (converted to AD within follow-up window). This project uses cross-sectional baseline labels only.
