# ============================================================
# 04_training.py — Train All 7 Models + 5-Fold CV
# ============================================================
# Industry-standard scaling applied consistently:
#   Features : [0,1] from dataset — no change
#   Targets  : load per-target MinMaxScaler saved by 03_hpo.py
#              Train & predict on [0,1] scaled targets
#              Inverse transform → physical units for reporting
#
# Outputs:
#   • Normalized metrics  (MSE, RMSE, MAE, R²) — comparable across models
#   • Physical-unit metrics (RMSE_phys, MAE_phys) — meaningful to readers
#   • 5-Fold CV with mean ± std
#   • All trained models saved as .joblib
#   • Prediction CSVs in both normalized and physical units
# ============================================================

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from config import (
    TARGETS, TARGET_META, RANDOM_SEED, N_FOLDS,
    DIR_SPLITS, DIR_FEATURES, DIR_HPO,
    DIR_MODELS, DIR_PREDICTIONS, DIR_METRICS,
    make_dirs
)

make_dirs()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true_01, y_pred_01, scaler):
    """
    Compute metrics on [0,1] scale and in physical units.
    scaler: fitted MinMaxScaler for this target.
    """
    mse  = mean_squared_error(y_true_01, y_pred_01)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true_01, y_pred_01)
    r2   = r2_score(y_true_01, y_pred_01)

    # Inverse transform to physical units
    y_true_phys = scaler.inverse_transform(y_true_01.reshape(-1,1)).ravel()
    y_pred_phys = scaler.inverse_transform(y_pred_01.reshape(-1,1)).ravel()

    rmse_phys = np.sqrt(mean_squared_error(y_true_phys, y_pred_phys))
    mae_phys  = mean_absolute_error(y_true_phys, y_pred_phys)

    return {
        "MSE"       : round(mse,       6),
        "RMSE"      : round(rmse,      6),
        "MAE"       : round(mae,       6),
        "R2"        : round(r2,        4),
        "RMSE_phys" : round(rmse_phys, 4),
        "MAE_phys"  : round(mae_phys,  4),
    }

def build_model(name, params):
    p  = dict(params)
    rs = RANDOM_SEED
    if name == "LinearRegression":
        return LinearRegression()
    elif name == "RandomForest":
        p.setdefault("random_state", rs); p.setdefault("n_jobs", -1)
        return RandomForestRegressor(**p)
    elif name == "SVR":
        return SVR(**p)
    elif name == "AdaBoost":
        p.setdefault("random_state", rs)
        return AdaBoostRegressor(**p)
    elif name == "XGBoost":
        p.setdefault("random_state", rs)
        p.setdefault("tree_method", "hist")
        p.setdefault("n_jobs", -1)
        return xgb.XGBRegressor(**p)
    elif name == "LightGBM":
        p.setdefault("random_state", rs)
        p.setdefault("verbose", -1)
        p.setdefault("n_jobs", -1)
        return lgb.LGBMRegressor(**p)
    elif name == "CatBoost":
        p.setdefault("random_seed", rs)
        p.setdefault("verbose", False)
        p.setdefault("task_type", "CPU")
        p.setdefault("loss_function", "RMSE")
        p.setdefault("iterations", 300)
        return CatBoostRegressor(**p)
    else:
        raise ValueError(f"Unknown model: {name}")

# ─────────────────────────────────────────────────────────────
# 1. Load Data, Scalers, HPO Params
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Data, Scalers & HPO Parameters")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DIR_SPLITS, "train.csv"))
val_df   = pd.read_csv(os.path.join(DIR_SPLITS, "val.csv"))
test_df  = pd.read_csv(os.path.join(DIR_SPLITS, "test.csv"))
for _df in [train_df, val_df, test_df]:
    _df.columns = _df.columns.str.strip().str.lower()

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

with open(os.path.join(DIR_FEATURES, "consensus_features.json")) as f:
    consensus_features = json.load(f)["consensus_features"]

with open(os.path.join(DIR_HPO, "best_params.json")) as f:
    best_params = json.load(f)

# ── Features: already [0,1] — use directly
X_train = train_df[consensus_features].values
X_val   = val_df[consensus_features].values
X_test  = test_df[consensus_features].values
X_full  = full_df[consensus_features].values

# ── Load per-target scalers (fitted on train in 03_hpo.py)
target_scalers = {}
for tgt in TARGETS:
    sname = tgt.replace(' ', '_')
    sc_path = os.path.join(DIR_MODELS, f"scaler_target_{sname}.joblib")
    target_scalers[tgt] = joblib.load(sc_path)
    print(f"   Loaded scaler for: {tgt}")

# ── Scale targets using loaded scalers (fitted on train only)
def scale_target(df, tgt):
    return target_scalers[tgt].transform(df[[tgt]].values).ravel()

print(f"\n   Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"   Features: {len(consensus_features)}")
print(f"   Models  : {list(best_params.keys())}")

# ─────────────────────────────────────────────────────────────
# 2. Train, Evaluate, 5-Fold CV
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Training & Evaluation")
print("-" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

test_records = []
val_records  = []
cv_records   = []

for model_name, target_params in best_params.items():
    print(f"\n▶ {model_name}")
    model_dir = os.path.join(DIR_MODELS, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # prediction dict: store both normalized and physical predictions
    pred_rows = {}
    for tgt in TARGETS:
        sc = target_scalers[tgt]
        pred_rows[f"{tgt}_actual_01"]    = scale_target(test_df, tgt)
        pred_rows[f"{tgt}_actual_phys"]  = test_df[tgt].values

    pred_rows["model"] = model_name

    for tgt in TARGETS:
        meta   = TARGET_META[tgt]
        sc     = target_scalers[tgt]
        params = target_params.get(tgt, {})

        # Scaled [0,1] targets
        y_train_01 = scale_target(train_df, tgt)
        y_val_01   = scale_target(val_df,   tgt)
        y_test_01  = scale_target(test_df,  tgt)
        y_full_01  = sc.transform(full_df[[tgt]].values).ravel()

        # ── Build & Train ──
        model = build_model(model_name, params)
        model.fit(X_train, y_train_01)

        # ── Save model ──
        joblib.dump(model, os.path.join(
            model_dir, f"{model_name}_{tgt.replace(' ','_')}.joblib"))

        # ── Test metrics ──
        y_pred_test_01 = model.predict(X_test)
        t_met = compute_metrics(y_test_01, y_pred_test_01, sc)
        t_met.update({"Model": model_name, "Target": tgt,
                      "Unit": meta["unit"], "Split": "Test"})
        test_records.append(t_met)

        # ── Val metrics ──
        y_pred_val_01 = model.predict(X_val)
        v_met = compute_metrics(y_val_01, y_pred_val_01, sc)
        v_met.update({"Model": model_name, "Target": tgt,
                      "Unit": meta["unit"], "Split": "Validation"})
        val_records.append(v_met)

        # ── 5-Fold CV ──
        cv_mse_raw = -cross_val_score(
            build_model(model_name, params),
            X_full, y_full_01,
            cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
        )
        cv_r2_raw = cross_val_score(
            build_model(model_name, params),
            X_full, y_full_01,
            cv=kf, scoring="r2", n_jobs=-1
        )
        # Physical RMSE from CV: sqrt(MSE_01) * physical_range
        phys_range = (sc.data_max_[0] - sc.data_min_[0])
        cv_records.append({
            "Model"              : model_name,
            "Target"             : tgt,
            "Unit"               : meta["unit"],
            "CV_MSE_mean"        : round(cv_mse_raw.mean(), 6),
            "CV_MSE_std"         : round(cv_mse_raw.std(),  6),
            "CV_RMSE_mean"       : round(np.sqrt(cv_mse_raw).mean(), 6),
            "CV_RMSE_std"        : round(np.sqrt(cv_mse_raw).std(),  6),
            "CV_R2_mean"         : round(cv_r2_raw.mean(), 4),
            "CV_R2_std"          : round(cv_r2_raw.std(),  4),
            "CV_RMSE_phys_mean"  : round(np.sqrt(cv_mse_raw).mean() * phys_range, 4),
            "CV_RMSE_phys_std"   : round(np.sqrt(cv_mse_raw).std()  * phys_range, 4),
        })

        # Store physical predictions
        pred_rows[f"{tgt}_pred_01"]   = y_pred_test_01
        pred_rows[f"{tgt}_pred_phys"] = sc.inverse_transform(
            y_pred_test_01.reshape(-1,1)).ravel()

        print(f"   {tgt:25s}  R²={t_met['R2']:.4f}  "
              f"RMSE={t_met['RMSE']:.5f}  "
              f"RMSE_phys={t_met['RMSE_phys']:.4f} {meta['unit']}")

    # ── Save predictions ──
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(DIR_PREDICTIONS, f"{model_name}_predictions.csv"),
        index=False
    )

# ─────────────────────────────────────────────────────────────
# 3. Save Metrics
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Saving Metrics")

test_out = pd.DataFrame(test_records)
val_out  = pd.DataFrame(val_records)
cv_out   = pd.DataFrame(cv_records)

test_out.to_csv(os.path.join(DIR_METRICS, "test_metrics.csv"),     index=False)
val_out.to_csv( os.path.join(DIR_METRICS, "val_metrics.csv"),      index=False)
cv_out.to_csv(  os.path.join(DIR_METRICS, "cv5fold_metrics.csv"),  index=False)

print(f"   test_metrics.csv    → {DIR_METRICS}")
print(f"   val_metrics.csv     → {DIR_METRICS}")
print(f"   cv5fold_metrics.csv → {DIR_METRICS}")

# ─────────────────────────────────────────────────────────────
# 4. Console Summary
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Test Set Summary (average across targets)")
avg = (test_out.groupby("Model")[["MSE","RMSE","MAE","R2"]]
       .mean().sort_values("R2", ascending=False))
print(avg.round(4).to_string())

print("\nPhysical-Unit RMSE Summary:")
phys = test_out.pivot_table(
    index="Model", columns="Target", values="RMSE_phys"
)
print(phys.round(4).to_string())

print("\n" + "=" * 60)
print("04_training.py COMPLETE")
print("=" * 60)