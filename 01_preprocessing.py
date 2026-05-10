# ============================================================
# 01_preprocessing.py — Data Loading, Splitting & EDA Plots
# ============================================================
# Changes vs original:
#   • Split FIRST (80/10/10) before any fitting → fixes leakage
#   • Saves exact train/val/test indices for reproducibility
#   • Feature variance analysis table (Issue 7)
#   • All plots saved as high-res PNG + PDF
#   • Bar values shown on all bar charts
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

from config import (
    DATA_FILE, TARGETS, TARGET_META,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    DIR_SPLITS, DIR_FEATURES, DIR_PLOTS,
    PLOT_DPI, PLOT_STYLE, FONT_SIZE, TITLE_SIZE, LABEL_SIZE,
    TARGET_COLORS, make_dirs
)

make_dirs()
plt.style.use(PLOT_STYLE)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(DIR_PLOTS, f"{name}.{ext}")
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {name}.png / .pdf")

def bold_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title,  fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SIZE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

def add_bar_values(ax, fmt="{:.3f}", fontsize=9, rotation=0):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h != 0:
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
                rotation=rotation,
                xytext=(0, 3), textcoords="offset points"
            )

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

# Build a rename map: strip + lower but DO NOT replace spaces
# so that "dielectric constant" stays "dielectric constant"
df.columns = df.columns.str.strip().str.lower()

# Verify all targets exist
missing = [t for t in TARGETS if t not in df.columns]
if missing:
    raise KeyError(f"Target columns not found in dataset: {missing}\n"
                   f"Available columns: {df.columns.tolist()}")

feature_cols = [c for c in df.columns if c not in TARGETS]

print(f"   Rows: {len(df):,}   |   Features: {len(feature_cols)}   |   Targets: {len(TARGETS)}")

# ─────────────────────────────────────────────────────────────
# 2. Train / Val / Test Split  ← FIRST, before any fitting
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Splitting (80 / 10 / 10)")

X = df[feature_cols]
y = df[TARGETS]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
)

print(f"   Train : {X_train.shape[0]:,} rows")
print(f"   Val   : {X_val.shape[0]:,} rows")
print(f"   Test  : {X_test.shape[0]:,} rows")

# Save indices for reproducibility (Issue 12)
np.save(os.path.join(DIR_SPLITS, "train_idx.npy"), X_train.index.to_numpy())
np.save(os.path.join(DIR_SPLITS, "val_idx.npy"),   X_val.index.to_numpy())
np.save(os.path.join(DIR_SPLITS, "test_idx.npy"),  X_test.index.to_numpy())

# Save split CSVs
train_df = pd.concat([X_train, y_train], axis=1)
val_df   = pd.concat([X_val,   y_val],   axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

train_df.to_csv(os.path.join(DIR_SPLITS, "train.csv"), index=False)
val_df.to_csv(  os.path.join(DIR_SPLITS, "val.csv"),   index=False)
test_df.to_csv( os.path.join(DIR_SPLITS, "test.csv"),  index=False)

split_info = {
    "train_rows": int(X_train.shape[0]),
    "val_rows"  : int(X_val.shape[0]),
    "test_rows" : int(X_test.shape[0]),
    "random_seed": RANDOM_SEED,
    "train_ratio": TRAIN_RATIO,
    "val_ratio"  : VAL_RATIO,
    "test_ratio" : TEST_RATIO,
}
with open(os.path.join(DIR_SPLITS, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

print("   Split indices and CSVs saved.")

# ─────────────────────────────────────────────────────────────
# 3. Summary Statistics
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Summary Statistics")

summary = df[feature_cols + TARGETS].describe().T
summary.to_csv(os.path.join(DIR_FEATURES, "summary_statistics.csv"))

# ── Plot: Mean ± Std with Min/Max markers ──
fig, ax = plt.subplots(figsize=(18, 7))
cols    = feature_cols + TARGETS
means   = df[cols].mean()
stds    = df[cols].std()
mins    = df[cols].min()
maxs    = df[cols].max()

x = np.arange(len(cols))
bars = ax.bar(x, means, yerr=stds, capsize=4,
              color="steelblue", edgecolor="black", alpha=0.82, width=0.6)
ax.scatter(x, mins, color="crimson",   marker="v", s=70, label="Min", zorder=5)
ax.scatter(x, maxs, color="darkgreen", marker="^", s=70, label="Max", zorder=5)

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + bar.get_height()*0.03,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(cols, rotation=90, fontsize=9, fontweight="bold")
bold_ax(ax, "Summary Statistics — BaTiO₃ Dataset",
        "Feature / Target", "Value")
ax.legend(fontsize=11)
fig.tight_layout()
save_fig(fig, "01_summary_statistics")

# ─────────────────────────────────────────────────────────────
# 4. Target Distributions
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Target Distributions")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, tgt in enumerate(TARGETS):
    meta   = TARGET_META[tgt]
    color  = TARGET_COLORS[tgt]
    data   = df[tgt].dropna()
    ax     = axes[i]

    ax.hist(data, bins=40, color=color, alpha=0.75,
            edgecolor="white", linewidth=0.5)

    # KDE overlay
    kde_x = np.linspace(data.min(), data.max(), 300)
    kde   = stats.gaussian_kde(data)
    ax2   = ax.twinx()
    ax2.plot(kde_x, kde(kde_x), color="black", linewidth=2, linestyle="--")
    ax2.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax2.tick_params(labelsize=9)

    # Stats annotation
    skw = data.skew()
    ax.axvline(data.mean(),   color="red",   linestyle="--", linewidth=1.5,
               label=f"Mean={data.mean():.3f}")
    ax.axvline(data.median(), color="navy",  linestyle=":",  linewidth=1.5,
               label=f"Median={data.median():.3f}")

    bold_ax(ax,
            f"{meta['symbol']} — {tgt.title()}\n"
            f"Skew={skw:.3f}  |  Std={data.std():.3f}",
            f"{tgt.title()} ({meta['unit']})", "Count")
    ax.legend(fontsize=9)

fig.suptitle("Target Property Distributions — BaTiO₃ Dataset",
             fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, "02_target_distributions")

# ─────────────────────────────────────────────────────────────
# 5. Correlation Heatmap
# ─────────────────────────────────────────────────────────────
print("\nSTEP 5 — Correlation Heatmap")

corr = df[feature_cols + TARGETS].corr()
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.4,
            annot_kws={"size": 7, "weight": "bold"},
            ax=ax, square=True)
bold_ax(ax, "Feature & Target Correlation Matrix", "", "")
plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="bold")
plt.yticks(rotation=0,  fontsize=9, fontweight="bold")
fig.tight_layout()
save_fig(fig, "03_correlation_heatmap")

# ─────────────────────────────────────────────────────────────
# 6. Feature Variance Analysis  (Issue 7)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 6 — Feature Variance Analysis")

variance_df = pd.DataFrame({
    "Feature"  : feature_cols,
    "Variance" : df[feature_cols].var().values,
    "Std"      : df[feature_cols].std().values,
    "Mean"     : df[feature_cols].mean().values,
    "CV_%"     : (df[feature_cols].std() / df[feature_cols].mean().replace(0, np.nan) * 100).values
}).sort_values("Variance", ascending=False).reset_index(drop=True)

variance_df.to_csv(os.path.join(DIR_FEATURES, "feature_variance.csv"), index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Variance bar
ax = axes[0]
colors = ["#d73027" if v < 0.01 else "#4575b4" for v in variance_df["Variance"]]
bars = ax.barh(variance_df["Feature"], variance_df["Variance"],
               color=colors, edgecolor="black", alpha=0.85)
for bar, val in zip(bars, variance_df["Variance"]):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8, fontweight="bold")
bold_ax(ax, "Feature Variance\n(Red = Low Variance < 0.01)", "Variance", "Feature")
ax.invert_yaxis()

# CV% bar
ax = axes[1]
bars = ax.barh(variance_df["Feature"], variance_df["CV_%"],
               color="#74add1", edgecolor="black", alpha=0.85)
for bar, val in zip(bars, variance_df["CV_%"]):
    if np.isfinite(val):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")
bold_ax(ax, "Coefficient of Variation (%)", "CV (%)", "Feature")
ax.invert_yaxis()

fig.suptitle("Feature Variance Analysis — Justification for Low Sintering-Param Importance",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save_fig(fig, "04_feature_variance_analysis")
print("   Saved feature_variance.csv")

# ─────────────────────────────────────────────────────────────
# 7. Pairplot — Targets Only
# ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — Target Pairplot")

pair_labels = {t: TARGET_META[t]["symbol"] for t in TARGETS}
df_pair = df[TARGETS].rename(columns=pair_labels)

g = sns.pairplot(df_pair, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 10},
                 diag_kws={"color": "steelblue"})
g.fig.suptitle("Pairplot of Target Properties — BaTiO₃", y=1.02,
               fontsize=14, fontweight="bold")

for ext in ("png", "pdf"):
    g.fig.savefig(os.path.join(DIR_PLOTS, f"05_target_pairplot.{ext}"),
                  dpi=PLOT_DPI, bbox_inches="tight")
plt.close("all")
print("   Saved 05_target_pairplot")

# ─────────────────────────────────────────────────────────────
# 8. Boxplots per Feature
# ─────────────────────────────────────────────────────────────
print("\nSTEP 8 — Feature Boxplots")

n_feat = len(feature_cols)
ncols  = 4
nrows  = int(np.ceil(n_feat / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="navy"),
                    medianprops=dict(color="red", linewidth=2))
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].tick_params(labelsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Feature Distributions — Boxplots", fontsize=14, fontweight="bold")
fig.tight_layout()
save_fig(fig, "06_feature_boxplots")

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("01_preprocessing.py COMPLETE")
print(f"   Splits  → {DIR_SPLITS}")
print(f"   Features→ {DIR_FEATURES}")
print(f"   Plots   → {DIR_PLOTS}")
print("=" * 60)