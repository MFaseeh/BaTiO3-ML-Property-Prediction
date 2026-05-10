# BaTiO₃ Multi-Property ML Prediction Framework

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Paper

> **"Multi-Model Machine Learning Framework for Predicting Material 
> Properties of BaTiO₃-Based Ferroelectric Ceramics"**  
> Saman Khalid, Muhammad Faseeh, Hyejeong Song, Murad Ali Khan,  
> Syed Shehryar Ali Naqvi, Hyunseok Ko, Do-Hyeun Kim  
> *Materials & Design*, 2025

## Overview

This repository contains the full source code for predicting 
four key functional properties of BaTiO₃ ferroelectric ceramics:

| Property | Symbol | Unit |
|---|---|---|
| Piezoelectric coefficient | d₃₃ | pC/N |
| Dielectric constant | εᵣ | dimensionless |
| Dielectric loss tangent | tanδ | ratio |
| Density | ρ | g/cm³ |

## Pipeline Structure

| Script | Description |
|---|---|
| `config.py` | Central configuration — paths, seeds, physical ranges |
| `run_pipeline.py` | Master runner — runs all steps in order |
| `01_preprocessing.py` | Data loading, 80/10/10 split, EDA plots |
| `02_feature_selection.py` | RF + SHAP consensus feature selection |
| `03_hpo.py` | Bayesian HPO via Optuna for all 7 models |
| `04_training.py` | Model training + 5-fold CV + metrics |
| `05_evaluation_plots.py` | All evaluation and comparison plots |
| `06_ablation.py` | Ablation studies |

## Models

| Category | Model |
|---|---|
| Baseline | Linear Regression |
| Baseline | Random Forest |
| Baseline | SVR |
| Gradient Boosting | AdaBoost |
| Gradient Boosting | XGBoost |
| Gradient Boosting | LightGBM |
| Gradient Boosting | **CatBoost** (best overall) |

## Key Results

| Property | R² | RMSE (physical units) |
|---|---|---|
| d₃₃ | 0.922 | 14.73 pC/N |
| εᵣ | 0.933 | 140.05 |
| tanδ | 0.855 | 0.0018 |
| ρ | 0.674 | 0.020 g/cm³ |

## Installation

```bash
git clone https://github.com/[username]/BaTiO3-ML-Property-Prediction
cd BaTiO3-ML-Property-Prediction
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline
python run_pipeline.py

# Run single step
python run_pipeline.py --step 1

# Run from a specific step
python run_pipeline.py --from 3
```

## Dataset

The experimental dataset is not publicly available due to 
institutional data governance policies.

To request access contact the corresponding authors:
- Prof. Do-Hyeun Kim — kimdh@jejunu.ac.kr

## Reproducibility

All experiments use fixed random seed 42. Key configuration 
parameters are centralized in `config.py`.


## License

MIT License — see `LICENSE` file for details.