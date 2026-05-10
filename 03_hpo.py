# ============================================================
# 03_hpo.py — Bayesian HPO with Optuna for ALL 7 Models
# ============================================================
# Industry-standard scaling:
#   Features : already [0,1] from dataset — kept as-is
#   Targets  : MinMaxScaler fitted on TRAIN only → [0,1]
#              Scaler objects saved for inverse transform
# All 7 models optimized on uniformly scaled [0,1] targets
# ============================================================

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from config import (
    TARGETS, RANDOM_SEED, N_TRIALS, HPO_CV_FOLDS,
    DIR_SPLITS, DIR_FEATURES, DIR_HPO, DIR_MODELS, make_dirs
)

make_dirs()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# 1. Load Training Data
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading & Scaling Training Data")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DIR_SPLITS, "train.csv"))
train_df.columns = train_df.columns.str.strip().str.lower()

with open(os.path.join(DIR_FEATURES, "consensus_features.json")) as f:
    feat_info = json.load(f)
consensus_features = feat_info["consensus_features"]

# ── Features: already [0,1] — use as-is
X_train = train_df[consensus_features].values

# ── Targets: scale each to [0,1] using MinMaxScaler on train only
target_scalers = {}
y_train_scaled = {}

os.makedirs(DIR_MODELS, exist_ok=True)

for tgt in TARGETS:
    sc = MinMaxScaler()
    y_scaled = sc.fit_transform(train_df[[tgt]].values).ravel()
    y_train_scaled[tgt] = y_scaled
    target_scalers[tgt] = sc
    sname = tgt.replace(' ', '_')
    joblib.dump(sc, os.path.join(DIR_MODELS, f"scaler_target_{sname}.joblib"))
    print(f"   {tgt:25s} → scaled [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")

print(f"\n   Train rows : {len(train_df):,}")
print(f"   Features   : {len(consensus_features)}")
print(f"   HPO trials : {N_TRIALS} per model-target pair")

cv = KFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# 2. Objective Functions (all on [0,1] targets)
# ─────────────────────────────────────────────────────────────

def _score(model, X, y):
    s = cross_val_score(model, X, y, cv=cv,
                        scoring="neg_mean_squared_error", n_jobs=-1)
    return -s.mean()

def obj_lr(trial, X, y):
    return _score(LinearRegression(), X, y)

def obj_rf(trial, X, y):
    p = {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"        : trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features"     : trial.suggest_categorical(
                                 "max_features", ["sqrt", "log2", 0.5, 0.7]),
        "random_state": RANDOM_SEED, "n_jobs": -1,
    }
    return _score(RandomForestRegressor(**p), X, y)

def obj_svr(trial, X, y):
    # Ranges calibrated for [0,1] scaled targets
    p = {
        "C"      : trial.suggest_float("C", 0.01, 10.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-4, 0.1, log=True),
        "gamma"  : trial.suggest_categorical("gamma", ["scale", "auto"]),
        "kernel" : trial.suggest_categorical("kernel", ["rbf", "linear"]),
    }
    return _score(SVR(**p), X, y)

def obj_ada(trial, X, y):
    p = {
        "n_estimators" : trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "random_state" : RANDOM_SEED,
    }
    return _score(AdaBoostRegressor(**p), X, y)

def obj_xgb(trial, X, y):
    p = {
        "n_estimators"    : trial.suggest_int("n_estimators", 50, 300),
        "max_depth"       : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "tree_method": "hist", "random_state": RANDOM_SEED, "n_jobs": -1,
    }
    return _score(xgb.XGBRegressor(**p), X, y)

def obj_lgb(trial, X, y):
    p = {
        "n_estimators"    : trial.suggest_int("n_estimators", 50, 300),
        "max_depth"       : trial.suggest_int("max_depth", 3, 12),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves"      : trial.suggest_int("num_leaves", 20, 80),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": RANDOM_SEED, "verbose": -1, "n_jobs": -1,
    }
    return _score(lgb.LGBMRegressor(**p), X, y)

def obj_cat(trial, X, y):
    p = {
        "iterations"   : trial.suggest_int("iterations", 100, 600),
        "depth"        : trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg"  : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_seed": RANDOM_SEED, "loss_function": "RMSE",
        "task_type": "CPU", "verbose": False,
    }
    return _score(CatBoostRegressor(**p), X, y)

# ─────────────────────────────────────────────────────────────
# 3. Model Registry
# ─────────────────────────────────────────────────────────────
MODEL_OBJECTIVES = {
    "LinearRegression": obj_lr,
    "RandomForest"    : obj_rf,
    "SVR"             : obj_svr,
    "AdaBoost"        : obj_ada,
    "XGBoost"         : obj_xgb,
    "LightGBM"        : obj_lgb,
    "CatBoost"        : obj_cat,
}
NO_HPO_MODELS = {"LinearRegression"}

# ─────────────────────────────────────────────────────────────
# 4. Run HPO
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Running HPO (targets scaled to [0,1])")
print("-" * 60)

all_best_params = {}
all_records     = []
total = len(MODEL_OBJECTIVES) * len(TARGETS)
done  = 0

for model_name, obj_fn in MODEL_OBJECTIVES.items():
    all_best_params[model_name] = {}

    for tgt in TARGETS:
        done += 1
        y = y_train_scaled[tgt]
        print(f"   [{done}/{total}] {model_name} → {tgt} ...", end=" ", flush=True)

        if model_name in NO_HPO_MODELS:
            best_params = {}
            best_val    = _score(LinearRegression(), X_train, y)
        else:
            study = optuna.create_study(
                direction="minimize",
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
            )
            study.optimize(
                lambda trial, X=X_train, y_=y: obj_fn(trial, X, y_),
                n_trials=N_TRIALS,
                n_jobs=1,
                show_progress_bar=False
            )
            best_params = study.best_trial.params
            best_val    = study.best_value

        all_best_params[model_name][tgt] = best_params

        record = {"Model": model_name, "Target": tgt,
                  "Best_MSE_CV_scaled": round(best_val, 6)}
        record.update(best_params)
        all_records.append(record)
        print(f"MSE(scaled)={best_val:.5f}")

# ─────────────────────────────────────────────────────────────
# 5. Save Results
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Saving HPO Results")

with open(os.path.join(DIR_HPO, "best_params.json"), "w") as f:
    json.dump(all_best_params, f, indent=2)

pd.DataFrame(all_records).to_csv(
    os.path.join(DIR_HPO, "hpo_summary.csv"), index=False
)

print(f"   best_params.json     → {DIR_HPO}")
print(f"   hpo_summary.csv      → {DIR_HPO}")
print(f"   target scalers (.joblib) → {DIR_MODELS}")

print("\n" + "=" * 60)
print("03_hpo.py COMPLETE")
print("=" * 60)