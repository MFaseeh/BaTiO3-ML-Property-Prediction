# ============================================================
# config.py — Central Configuration for BaTiO3 ML Pipeline
# ============================================================

import os

# ── Random seed ──────────────────────────────────────────────
RANDOM_SEED = 42

# ── Dataset ──────────────────────────────────────────────────
DATA_FILE = "BA_expanded_zero_formula_coolrate_fixed.csv"

# ── Targets ──────────────────────────────────────────────────
TARGETS = ["d33", "dielectric constant", "tangent loss", "density"]

TARGET_META = {
    "d33": {
        "symbol": r"$d_{33}$",
        "unit": "pC/N",
        "phys_min": 124.012,
        "phys_max": 419.432,
    },
    "dielectric constant": {
        "symbol": r"$\varepsilon_r$",
        "unit": "dimensionless",
        "phys_min": 1298.68,
        "phys_max": 4208.71,
    },
    "tangent loss": {
        "symbol": r"$\tan\delta$",
        "unit": "ratio",
        "phys_min": 0.00334,
        "phys_max": 0.03367,
    },
    "density": {
        "symbol": r"$\rho$",
        "unit": r"g cm$^{-3}$",
        "phys_min": 5.618,
        "phys_max": 5.885,
    },
}

# ── Split ratios ─────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Cross-validation ─────────────────────────────────────────
N_FOLDS = 5

# ── HPO ──────────────────────────────────────────────────────
N_TRIALS      = 50          # Optuna trials per model-target pair
HPO_CV_FOLDS  = 3

# ── Output directories ────────────────────────────────────────
DIR_SPLITS      = "outputs/splits"
DIR_HPO         = "outputs/hpo"
DIR_MODELS      = "outputs/models"
DIR_PREDICTIONS = "outputs/predictions"
DIR_METRICS     = "outputs/metrics"
DIR_PLOTS       = "outputs/plots"
DIR_SHAP        = "outputs/shap"
DIR_FEATURES    = "outputs/features"

ALL_DIRS = [
    DIR_SPLITS, DIR_HPO, DIR_MODELS,
    DIR_PREDICTIONS, DIR_METRICS,
    DIR_PLOTS, DIR_SHAP, DIR_FEATURES
]

def make_dirs():
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────
PLOT_DPI    = 300
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
FONT_SIZE   = 12
TITLE_SIZE  = 14
LABEL_SIZE  = 12

# ── Colour palette ────────────────────────────────────────────
MODEL_COLORS = {
    "LinearRegression" : "#4C72B0",
    "RandomForest"     : "#55A868",
    "SVR"              : "#C44E52",
    "AdaBoost"         : "#8172B2",
    "XGBoost"          : "#CCB974",
    "LightGBM"         : "#64B5CD",
    "CatBoost"         : "#E07B54",
}

TARGET_COLORS = {
    "d33"                : "#2196F3",
    "dielectric constant": "#4CAF50",
    "tangent loss"       : "#FF9800",
    "density"            : "#9C27B0",
}
