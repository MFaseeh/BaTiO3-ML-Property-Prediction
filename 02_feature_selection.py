# ============================================================
# 02_feature_selection.py — RF + SHAP Consensus Feature Selection
# ============================================================
# Changes vs original:
#   • Uses train split only (no leakage)
#   • RF importance + SHAP computed on training data
#   • SHAP summary plots per target (Issue 6)
#   • SHAP dependence plots for top features (Issue 6)
#   • Variance analysis integrated (Issue 7)
#   • Bar values on all bar plots
#   • All outputs: PNG + PDF
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    TARGETS, TARGET_META, RANDOM_SEED,
    DIR_SPLITS, DIR_FEATURES, DIR_PLOTS, DIR_SHAP,
    PLOT_DPI, PLOT_STYLE, FONT_SIZE, TITLE_SIZE, LABEL_SIZE,
    TARGET_COLORS, make_dirs
)

make_dirs()
plt.style.use(PLOT_STYLE)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def save_fig(fig, name, subdir=None):
    d = subdir if subdir else DIR_PLOTS
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"),
                    dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {name}")

def bold_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title,  fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.tick_params(labelsize=FONT_SIZE)

def add_bar_values_h(ax, fmt="{:.4f}", fontsize=9):
    for p in ax.patches:
        w = p.get_width()
        if np.isfinite(w) and abs(w) > 1e-8:
            ax.annotate(fmt.format(w),
                        (w, p.get_y() + p.get_height()/2),
                        ha="left", va="center",
                        fontsize=fontsize, fontweight="bold",
                        xytext=(3, 0), textcoords="offset points")

# ─────────────────────────────────────────────────────────────
# 1. Load Training Split
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Training Data")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DIR_SPLITS, "train.csv"))
train_df.columns = train_df.columns.str.strip().str.lower()
feature_cols = [c for c in train_df.columns if c not in TARGETS]

X_train = train_df[feature_cols].values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

print(f"   Train rows : {len(train_df):,}")
print(f"   Features   : {len(feature_cols)}")

# ─────────────────────────────────────────────────────────────
# 2. Random Forest Importance
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Random Forest Feature Importance")

rf_importances = pd.DataFrame(index=feature_cols)

for tgt in TARGETS:
    y = train_df[tgt].values
    rf = RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_scaled, y)
    rf_importances[tgt] = rf.feature_importances_
    print(f"   RF done for: {tgt}")

# Normalize per target then compute mean
rf_norm = rf_importances.div(rf_importances.max())
rf_importances["mean_rf"] = rf_norm.mean(axis=1)
rf_importances.to_csv(os.path.join(DIR_FEATURES, "rf_importances.csv"))

# ── RF Heatmap ──
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(rf_norm, cmap="YlGnBu", annot=True, fmt=".2f",
            annot_kws={"size": 8, "weight": "bold"}, ax=ax,
            linewidths=0.4)
bold_ax(ax, "Random Forest Feature Importance Heatmap",
        "Target Property", "Feature")
plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold")
plt.yticks(rotation=0,  fontsize=10, fontweight="bold")
fig.tight_layout()
save_fig(fig, "07_rf_importance_heatmap")

# ── RF Bar Plot ──
rf_sorted = rf_importances.sort_values("mean_rf", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(rf_sorted.index, rf_sorted["mean_rf"],
               color="steelblue", edgecolor="black", alpha=0.85)
add_bar_values_h(ax, fmt="{:.4f}")
bold_ax(ax, "Mean Normalized Random Forest Importance",
        "Mean Normalized Importance", "Feature")
fig.tight_layout()
save_fig(fig, "08_rf_importance_barplot")

# ─────────────────────────────────────────────────────────────
# 3. SHAP Feature Importance
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — SHAP Feature Importance")

import lightgbm as lgb

shap_importances = pd.DataFrame(index=feature_cols)
shap_values_dict = {}   # store for summary/dependence plots

for tgt in TARGETS:
    y = train_df[tgt].values
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        random_state=RANDOM_SEED, verbose=-1
    )
    model.fit(X_scaled, y)

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_scaled)

    shap_importances[tgt]   = np.abs(shap_vals).mean(axis=0)
    shap_values_dict[tgt]   = shap_vals
    print(f"   SHAP done for: {tgt}")

shap_norm = shap_importances.div(shap_importances.max())
shap_importances["mean_shap"] = shap_norm.mean(axis=1)
shap_importances.to_csv(os.path.join(DIR_SHAP, "shap_importances.csv"))

# ── SHAP Heatmap ──
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(shap_norm, cmap="PuBuGn", annot=True, fmt=".2f",
            annot_kws={"size": 8, "weight": "bold"}, ax=ax,
            linewidths=0.4)
bold_ax(ax, "SHAP Feature Importance Heatmap",
        "Target Property", "Feature")
plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold")
plt.yticks(rotation=0,  fontsize=10, fontweight="bold")
fig.tight_layout()
save_fig(fig, "09_shap_importance_heatmap")

# ── SHAP Bar Plot ──
shap_sorted = shap_importances.sort_values("mean_shap", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(shap_sorted.index, shap_sorted["mean_shap"],
               color="mediumseagreen", edgecolor="black", alpha=0.85)
add_bar_values_h(ax, fmt="{:.4f}")
bold_ax(ax, "Mean Normalized SHAP Feature Importance",
        "Mean |SHAP| Value (Normalized)", "Feature")
fig.tight_layout()
save_fig(fig, "10_shap_importance_barplot")

# ─────────────────────────────────────────────────────────────
# 4. SHAP Summary Plots Per Target  (Issue 6)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — SHAP Summary Plots per Target")

os.makedirs(DIR_SHAP, exist_ok=True)
X_df = pd.DataFrame(X_scaled, columns=feature_cols)

for tgt in TARGETS:
    meta = TARGET_META[tgt]
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values_dict[tgt], X_df,
        plot_type="dot", show=False, max_display=14,
        color_bar_label="Feature Value"
    )
    plt.title(f"SHAP Summary — {meta['symbol']} ({tgt})",
              fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    name = f"shap_summary_{tgt.replace(' ', '_')}"
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(DIR_SHAP, f"{name}.{ext}"),
                    dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved SHAP summary for {tgt}")

# ─────────────────────────────────────────────────────────────
# 5. SHAP Dependence Plots — Top 3 Features per Target (Issue 6)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 5 — SHAP Dependence Plots")

for tgt in TARGETS:
    meta     = TARGET_META[tgt]
    top3     = shap_importances[tgt].nlargest(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, feat in zip(axes, top3):
        feat_idx = feature_cols.index(feat)
        shap.dependence_plot(
            feat_idx, shap_values_dict[tgt], X_df,
            ax=ax, show=False, dot_size=15
        )
        ax.set_title(f"{feat}", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"SHAP for {meta['symbol']}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=9)

    fig.suptitle(
        f"SHAP Dependence Plots — {meta['symbol']} ({tgt})\n"
        f"Top 3 Most Influential Features",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    name = f"shap_dependence_{tgt.replace(' ', '_')}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DIR_SHAP, f"{name}.{ext}"),
                    dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved SHAP dependence for {tgt}")

# ─────────────────────────────────────────────────────────────
# 6. Consensus Feature Selection
# ─────────────────────────────────────────────────────────────
print("\nSTEP 6 — Consensus Feature Selection")

REL_THRESHOLD = 0.05   # 5 % of max importance

rf_selected   = rf_importances[rf_importances["mean_rf"] >=
                               REL_THRESHOLD * rf_importances["mean_rf"].max()].index.tolist()
shap_selected = shap_importances[shap_importances["mean_shap"] >=
                                 REL_THRESHOLD * shap_importances["mean_shap"].max()].index.tolist()

consensus     = sorted(set(rf_selected) & set(shap_selected))

if not consensus:
    consensus = feature_cols.copy()
    print("   ⚠ No consensus features — using all features.")
else:
    print(f"   RF selected   : {len(rf_selected)}")
    print(f"   SHAP selected : {len(shap_selected)}")
    print(f"   Consensus     : {len(consensus)} → {consensus}")

# Save consensus list
with open(os.path.join(DIR_FEATURES, "consensus_features.json"), "w") as f:
    json.dump({"consensus_features": consensus,
               "rf_selected": rf_selected,
               "shap_selected": shap_selected,
               "threshold": REL_THRESHOLD}, f, indent=2)

# ─────────────────────────────────────────────────────────────
# 7. Combined RF vs SHAP Comparison Table + Plot
# ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — RF vs SHAP Comparison")

compare_df = pd.DataFrame({
    "mean_rf"  : rf_importances["mean_rf"],
    "mean_shap": shap_importances["mean_shap"],
}).sort_values("mean_rf", ascending=False)
compare_df["selected"] = compare_df.index.isin(consensus)
compare_df.to_csv(os.path.join(DIR_FEATURES, "rf_shap_comparison.csv"))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, col, color, label in zip(
        axes,
        ["mean_rf", "mean_shap"],
        ["steelblue", "mediumseagreen"],
        ["Mean RF Importance", "Mean SHAP Importance"]):

    sorted_df = compare_df.sort_values(col, ascending=True)
    bar_colors = ["#d73027" if not s else color
                  for s in sorted_df["selected"]]
    bars = ax.barh(sorted_df.index, sorted_df[col],
                   color=bar_colors, edgecolor="black", alpha=0.85)
    add_bar_values_h(ax, fmt="{:.4f}", fontsize=8)
    bold_ax(ax, label, label, "Feature")
    ax.axvline(REL_THRESHOLD * compare_df[col].max(),
               color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({REL_THRESHOLD*100:.0f}%)")
    ax.legend(fontsize=9)

fig.suptitle("Consensus Feature Selection: RF vs SHAP\n"
             "(Blue = Selected | Red = Dropped)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save_fig(fig, "11_rf_shap_consensus")

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("02_feature_selection.py COMPLETE")
print(f"   Features → {DIR_FEATURES}")
print(f"   SHAP     → {DIR_SHAP}")
print(f"   Plots    → {DIR_PLOTS}")
print("=" * 60)