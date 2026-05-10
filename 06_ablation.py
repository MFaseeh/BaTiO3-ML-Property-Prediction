# ============================================================
# 06_ablation.py — Ablation Studies
# ============================================================
# Study 1: With vs Without Domain-Specific Features
# Study 2: With vs Without CTGAN Augmentation (proxy)
# Study 3: With vs Without Feature Selection (consensus vs all)
# ============================================================

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from config import (
    TARGETS, TARGET_META, RANDOM_SEED,
    DIR_SPLITS, DIR_FEATURES, DIR_HPO, DIR_METRICS, DIR_PLOTS, DIR_MODELS,
    PLOT_DPI, PLOT_STYLE, FONT_SIZE, TITLE_SIZE, LABEL_SIZE,
    MODEL_COLORS, make_dirs
)

make_dirs()
plt.style.use(PLOT_STYLE)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DIR_PLOTS, f"{name}.{ext}"),
                    dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {name}")

def bold_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title,   fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.tick_params(labelsize=FONT_SIZE)

def add_bar_values(ax, fmt="{:.4f}", fontsize=8):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and abs(h) > 1e-9:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom",
                        fontsize=fontsize, fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")

def quick_eval(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MSE":round(mse,6),"RMSE":round(rmse,6),
            "MAE":round(mae,6),"R2":round(r2,4)}

def build_model(name, params):
    p  = dict(params)
    rs = RANDOM_SEED
    if name == "LinearRegression" : return LinearRegression()
    if name == "RandomForest"     : p.setdefault("random_state",rs); p.setdefault("n_jobs",-1); return RandomForestRegressor(**p)
    if name == "SVR"              : return SVR(**p)
    if name == "AdaBoost"         : p.setdefault("random_state",rs); return AdaBoostRegressor(**p)
    if name == "XGBoost"          : p.setdefault("random_state",rs); p.setdefault("tree_method","hist"); p.setdefault("n_jobs",-1); return xgb.XGBRegressor(**p)
    if name == "LightGBM"         : p.setdefault("random_state",rs); p.setdefault("verbose",-1); p.setdefault("n_jobs",-1); return lgb.LGBMRegressor(**p)
    if name == "CatBoost"         : p.setdefault("random_seed",rs); p.setdefault("verbose",False); p.setdefault("task_type","CPU"); p.setdefault("loss_function","RMSE"); p.setdefault("iterations",300); return CatBoostRegressor(**p)
    raise ValueError(f"Unknown: {name}")

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Data for Ablation Studies")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DIR_SPLITS, "train.csv"))
val_df   = pd.read_csv(os.path.join(DIR_SPLITS, "val.csv"))
test_df  = pd.read_csv(os.path.join(DIR_SPLITS, "test.csv"))
for _df in [train_df, val_df, test_df]:
    _df.columns = _df.columns.str.strip().str.lower()

with open(os.path.join(DIR_FEATURES, "consensus_features.json")) as f:
    feat_info = json.load(f)
consensus_features = feat_info["consensus_features"]
all_features       = [c for c in train_df.columns if c not in TARGETS]

with open(os.path.join(DIR_HPO, "best_params.json")) as f:
    best_params = json.load(f)

# Load per-target scalers (fitted on train only in 03_hpo.py)
target_scalers = {}
for tgt in TARGETS:
    sname   = tgt.replace(" ", "_")
    sc_path = os.path.join(DIR_MODELS, f"scaler_target_{sname}.joblib")
    target_scalers[tgt] = joblib.load(sc_path)

# Scale targets using train-fitted scalers
for tgt in TARGETS:
    train_df[tgt] = target_scalers[tgt].transform(train_df[[tgt]].values).ravel()
    val_df[tgt]   = target_scalers[tgt].transform(val_df[[tgt]].values).ravel()
    test_df[tgt]  = target_scalers[tgt].transform(test_df[[tgt]].values).ravel()

print("   Targets scaled to [0,1] using train-fitted scalers.")

# Focus models for ablation (top 3 boosting + 1 baseline)
ABLATION_MODELS = ["LightGBM", "XGBoost", "CatBoost", "RandomForest"]

# ─────────────────────────────────────────────────────────────
# Study 1: With vs Without Domain-Specific Features
# ─────────────────────────────────────────────────────────────
print("\nSTUDY 1 — With vs Without Domain-Specific Features")

# Domain-specific features (synthesised)
DOMAIN_FEATURES = ["ba_ratio", "ti_ratio", "o_ratio",
                   "a_site_dopant", "b_site_dopant", "cool_rate",
                   "chemical formula"]
DOMAIN_FEATURES = [f for f in DOMAIN_FEATURES if f in all_features]

# Without domain features
no_domain_features = [f for f in consensus_features
                      if f not in DOMAIN_FEATURES]
if not no_domain_features:
    no_domain_features = [f for f in all_features if f not in DOMAIN_FEATURES]

records_s1 = []

for feat_set, label in [(consensus_features, "With Domain Features"),
                         (no_domain_features, "Without Domain Features")]:
    sc = StandardScaler()
    Xtr = sc.fit_transform(train_df[feat_set].values)
    Xte = sc.transform(test_df[feat_set].values)

    for mname in ABLATION_MODELS:
        params = best_params.get(mname, {})
        for tgt in TARGETS:
            p = params.get(tgt, {})
            model = build_model(mname, p)
            model.fit(Xtr, train_df[tgt].values)
            metrics = quick_eval(test_df[tgt].values, model.predict(Xte))
            metrics.update({"Model":mname, "Target":tgt, "Setting":label})
            records_s1.append(metrics)
            print(f"   {label[:5]} | {mname:15s} | {tgt:20s} R²={metrics['R2']:.4f}")

df_s1 = pd.DataFrame(records_s1)
df_s1.to_csv(os.path.join(DIR_METRICS, "ablation_domain_features.csv"), index=False)

# Plot
avg_s1 = df_s1.groupby(["Model","Setting"])["R2"].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Model", y="R2", hue="Setting", data=avg_s1,
            palette=["#2196F3","#FF5722"], ax=ax, alpha=0.88,
            order=ABLATION_MODELS)
add_bar_values(ax)
bold_ax(ax, "Ablation Study 1: Domain-Specific Feature Synthesis\nAverage R² across all targets",
        "Model", "Average R²")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)
fig.tight_layout()
save_fig(fig, "ablation_study1_domain_features")

# ─────────────────────────────────────────────────────────────
# Study 2: With Consensus FS vs All Features
# ─────────────────────────────────────────────────────────────
print("\nSTUDY 2 — Consensus FS vs All Features")

records_s2 = []

for feat_set, label in [(consensus_features, "With Feature Selection"),
                         (all_features,       "Without Feature Selection")]:
    sc  = StandardScaler()
    Xtr = sc.fit_transform(train_df[feat_set].values)
    Xte = sc.transform(test_df[feat_set].values)

    for mname in ABLATION_MODELS:
        params = best_params.get(mname, {})
        for tgt in TARGETS:
            p     = params.get(tgt, {})
            model = build_model(mname, p)
            model.fit(Xtr, train_df[tgt].values)
            metrics = quick_eval(test_df[tgt].values, model.predict(Xte))
            metrics.update({"Model":mname, "Target":tgt, "Setting":label})
            records_s2.append(metrics)
            print(f"   {label[:5]} | {mname:15s} | {tgt:20s} R²={metrics['R2']:.4f}")

df_s2 = pd.DataFrame(records_s2)
df_s2.to_csv(os.path.join(DIR_METRICS, "ablation_feature_selection.csv"), index=False)

avg_s2 = df_s2.groupby(["Model","Setting"])["R2"].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Model", y="R2", hue="Setting", data=avg_s2,
            palette=["#4CAF50","#FF9800"], ax=ax, alpha=0.88,
            order=ABLATION_MODELS)
add_bar_values(ax)
bold_ax(ax, "Ablation Study 2: Consensus Feature Selection\nAverage R² across all targets",
        "Model", "Average R²")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)
fig.tight_layout()
save_fig(fig, "ablation_study2_feature_selection")

# ─────────────────────────────────────────────────────────────
# Study 3: Gradient Boosting vs Baseline Models
# ─────────────────────────────────────────────────────────────
print("\nSTUDY 3 — Gradient Boosting vs Baselines")

sc  = StandardScaler()
Xtr = sc.fit_transform(train_df[consensus_features].values)
Xte = sc.transform(test_df[consensus_features].values)

records_s3 = []
for mname, mparams in best_params.items():
    for tgt in TARGETS:
        p     = mparams.get(tgt, {})
        model = build_model(mname, p)
        model.fit(Xtr, train_df[tgt].values)
        metrics = quick_eval(test_df[tgt].values, model.predict(Xte))
        metrics.update({
            "Model"    : mname,
            "Target"   : tgt,
            "Category" : ("Baseline" if mname in
                          ["LinearRegression","RandomForest","SVR"]
                          else "Gradient Boosting")
        })
        records_s3.append(metrics)

df_s3 = pd.DataFrame(records_s3)
df_s3.to_csv(os.path.join(DIR_METRICS, "ablation_baseline_vs_boosting.csv"), index=False)

avg_s3 = df_s3.groupby(["Model","Category"])["R2"].mean().reset_index()
avg_s3 = avg_s3.sort_values("R2", ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
palette = {"Baseline": "#78909C", "Gradient Boosting": "#1565C0"}
colors  = [palette[c] for c in avg_s3["Category"]]
bars    = ax.bar(avg_s3["Model"], avg_s3["R2"],
                 color=colors, edgecolor="black", alpha=0.88, width=0.6)
for bar, val in zip(bars, avg_s3["R2"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5,
           label="R²=0.90 threshold")
bold_ax(ax, "Ablation Study 3: Gradient Boosting vs Baseline Models\nAverage R² across all targets",
        "Model", "Average R²")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)

# Category legend
patches = [plt.Rectangle((0,0),1,1, color=v, label=k)
           for k,v in palette.items()]
ax.legend(handles=patches + [plt.Line2D([0],[0],color="red",
          linestyle="--", label="R²=0.90")], fontsize=10)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
save_fig(fig, "ablation_study3_boosting_vs_baseline")

# ─────────────────────────────────────────────────────────────
# Combined Ablation Summary Table
# ─────────────────────────────────────────────────────────────
print("\nSTEP — Saving Combined Ablation Summary")

summary_rows = []
for df_, study in [(df_s1, "Domain Features"),
                   (df_s2, "Feature Selection"),
                   (df_s3, "Model Category")]:
    avg = df_.groupby(["Model","Setting"] if "Setting" in df_.columns
                      else ["Model","Category"])[["R2","RMSE","MAE"]].mean()
    avg["Study"] = study
    summary_rows.append(avg.reset_index())

pd.concat(summary_rows).to_csv(
    os.path.join(DIR_METRICS, "ablation_combined_summary.csv"), index=False
)

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("06_ablation.py COMPLETE")
print("=" * 60)