# ============================================================
# 05_evaluation_plots.py — All Evaluation & Comparison Plots
# ============================================================
# Generates:
#   • Scatter (predicted vs actual) per model per target
#   • Residual distribution per model per target
#   • Residual boxplot across all models
#   • Metric comparison bars (MSE, RMSE, MAE, R²) — with values
#   • 5-Fold CV comparison (mean ± std)
#   • De-normalized metrics bar chart
#   • Heatmap of R² across models × targets
#   • Error distribution violin plots
#   • Model ranking summary chart
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from config import (
    TARGETS, TARGET_META, RANDOM_SEED,
    DIR_PREDICTIONS, DIR_METRICS, DIR_PLOTS,
    PLOT_DPI, PLOT_STYLE, FONT_SIZE, TITLE_SIZE, LABEL_SIZE,
    MODEL_COLORS, TARGET_COLORS, make_dirs
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

def add_bar_values(ax, fmt="{:.4f}", fontsize=8, rotation=0, pad=2):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and abs(h) > 1e-9:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom",
                        fontsize=fontsize, fontweight="bold",
                        rotation=rotation,
                        xytext=(0, pad), textcoords="offset points")

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Metrics & Predictions")
print("=" * 60)

test_metrics = pd.read_csv(os.path.join(DIR_METRICS, "test_metrics.csv"))
cv_metrics   = pd.read_csv(os.path.join(DIR_METRICS, "cv5fold_metrics.csv"))
models       = test_metrics["Model"].unique().tolist()

print(f"   Models : {models}")

# Load all predictions
preds = {}
for m in models:
    fp = os.path.join(DIR_PREDICTIONS, f"{m}_predictions.csv")
    if os.path.exists(fp):
        preds[m] = pd.read_csv(fp)

# ─────────────────────────────────────────────────────────────
# 2. Scatter Plots — Predicted vs Actual (per model, per target)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Scatter Plots")

for m in models:
    if m not in preds:
        continue
    df_pred = preds[m]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    color = MODEL_COLORS.get(m, "steelblue")

    for i, tgt in enumerate(TARGETS):
        ax   = axes[i]
        meta = TARGET_META[tgt]
        ya   = df_pred[f"{tgt}_actual_phys"].values
        yp   = df_pred[f"{tgt}_pred_phys"].values

        ax.scatter(ya, yp, color=color, alpha=0.55, s=18, edgecolors="none")

        mn = min(ya.min(), yp.min())
        mx = max(ya.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.8, label="Ideal (y=x)")

        # Regression line
        slope, intercept, r, p, _ = stats.linregress(ya, yp)
        x_line = np.linspace(mn, mx, 200)
        ax.plot(x_line, slope*x_line + intercept, "b-", linewidth=1.2,
                label=f"Fit (r={r:.3f})")

        r2_val = test_metrics.loc[
            (test_metrics["Model"]==m) & (test_metrics["Target"]==tgt), "R2"
        ].values
        r2_str = f"R²={r2_val[0]:.4f}" if len(r2_val) else ""

        bold_ax(ax,
                f"{m} — {meta['symbol']}\n{r2_str}",
                f"Actual ({meta['unit']})",
                f"Predicted ({meta['unit']})")
        ax.legend(fontsize=9)

    fig.suptitle(f"{m} — Predicted vs Actual (All Targets)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"scatter_{m}")

# ─────────────────────────────────────────────────────────────
# 3. Residual Distributions — Per Model, Per Target
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3 — Residual Distribution Plots")

for m in models:
    if m not in preds:
        continue
    df_pred = preds[m]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    color = MODEL_COLORS.get(m, "steelblue")

    for i, tgt in enumerate(TARGETS):
        ax        = axes[i]
        meta      = TARGET_META[tgt]
        residuals = df_pred[f"{tgt}_actual_phys"].values - df_pred[f"{tgt}_pred_phys"].values

        ax.hist(residuals, bins=35, color=color, alpha=0.72,
                edgecolor="white", linewidth=0.4, density=True)

        # KDE overlay
        kde_x = np.linspace(residuals.min(), residuals.max(), 300)
        kde   = stats.gaussian_kde(residuals)
        ax.plot(kde_x, kde(kde_x), "k-", linewidth=2)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)

        bold_ax(ax,
                f"{m} — {meta['symbol']}\n"
                f"μ={residuals.mean():.4f}  σ={residuals.std():.4f}",
                f"Residual (norm.)", "Density")

    fig.suptitle(f"{m} — Residual Distributions (All Targets)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"residuals_{m}")

# ─────────────────────────────────────────────────────────────
# 4. Residual Boxplot — All Models Combined
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4 — Combined Residual Boxplot")

resid_rows = []
for m in models:
    if m not in preds:
        continue
    df_pred = preds[m]
    for tgt in TARGETS:
        resid = df_pred[f"{tgt}_actual_phys"].values - df_pred[f"{tgt}_pred_phys"].values
        for r in resid:
            resid_rows.append({"Model": m, "Target": tgt, "Residual": r})

resid_df = pd.DataFrame(resid_rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# By model
ax = axes[0]
palette = [MODEL_COLORS.get(m, "#aaa") for m in models]
sns.boxplot(x="Model", y="Residual", data=resid_df,
            palette=palette, ax=ax, width=0.5,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
bold_ax(ax, "Residual Distribution by Model", "Model", "Residual (physical units)")
ax.tick_params(axis="x", rotation=30)

# By target
ax = axes[1]
t_palette = [TARGET_COLORS[t] for t in TARGETS]
sns.boxplot(x="Target", y="Residual", data=resid_df,
            palette=t_palette, ax=ax, width=0.5,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
bold_ax(ax, "Residual Distribution by Target", "Target Property", "Residual (physical units)")
ax.tick_params(axis="x", rotation=30)

fig.suptitle("Residual Analysis — All Models & Targets",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "residual_boxplot_combined")

# ─────────────────────────────────────────────────────────────
# 5. Metric Comparison Bar Charts (MSE, RMSE, MAE, R²)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 5 — Metric Comparison Bar Charts")

for metric in ["MSE", "RMSE", "MAE", "R2"]:
    fig, ax = plt.subplots(figsize=(14, 7))
    palette = [TARGET_COLORS[t] for t in TARGETS]
    bars = sns.barplot(
        x="Model", y=metric, hue="Target",
        data=test_metrics,
        palette=palette, ax=ax, alpha=0.88,
        order=models
    )
    add_bar_values(ax, fmt="{:.4f}", fontsize=7, rotation=60, pad=1)
    bold_ax(ax,
            f"Model Comparison — {metric} (Test Set)",
            "Model", metric)
    ax.legend(title="Target", fontsize=9,
              title_fontsize=10, framealpha=0.9)
    if metric == "R2":
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, f"comparison_{metric}")

# ─────────────────────────────────────────────────────────────
# 6. 5-Fold CV Comparison — R² Mean ± Std
# ─────────────────────────────────────────────────────────────
print("\nSTEP 6 — 5-Fold CV Comparison")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, tgt in enumerate(TARGETS):
    ax     = axes[i]
    meta   = TARGET_META[tgt]
    subset = cv_metrics[cv_metrics["Target"] == tgt].copy()
    subset = subset.sort_values("CV_R2_mean", ascending=False)

    colors = [MODEL_COLORS.get(m, "#aaa") for m in subset["Model"]]
    bars = ax.bar(subset["Model"], subset["CV_R2_mean"],
                  yerr=subset["CV_R2_std"], capsize=5,
                  color=colors, edgecolor="black", alpha=0.85, width=0.6)

    for bar, mean, std in zip(bars, subset["CV_R2_mean"], subset["CV_R2_std"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.005,
                f"{mean:.3f}±{std:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.9, color="red", linestyle="--", linewidth=1.2,
               label="R²=0.90 threshold")
    ax.set_ylim(0, 1.1)
    bold_ax(ax,
            f"5-Fold CV R² — {meta['symbol']}",
            "Model", "R² (mean ± std)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)

fig.suptitle("5-Fold Cross-Validation R² — All Models & Targets",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "cv5fold_r2_comparison")

# ─────────────────────────────────────────────────────────────
# 7. De-normalized Metrics Chart (Physical Units)  (Issue 2)
# ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — De-normalized Metrics (Physical Units)")

fig, axes = plt.subplots(1, len(TARGETS), figsize=(20, 7))

for i, tgt in enumerate(TARGETS):
    ax     = axes[i]
    meta   = TARGET_META[tgt]
    subset = test_metrics[test_metrics["Target"] == tgt].sort_values(
        "RMSE_phys", ascending=True
    )
    colors = [MODEL_COLORS.get(m, "#aaa") for m in subset["Model"]]
    bars = ax.barh(subset["Model"], subset["RMSE_phys"],
                   color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, subset["RMSE_phys"]):
        ax.text(bar.get_width() + bar.get_width()*0.02,
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center",
                fontsize=9, fontweight="bold")

    bold_ax(ax,
            f"{meta['symbol']}\nRMSE ({meta['unit']})",
            f"RMSE [{meta['unit']}]", "Model")
    ax.invert_yaxis()

fig.suptitle("De-normalized RMSE in Physical Units — Test Set",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "denormalized_rmse_physical_units")

# ─────────────────────────────────────────────────────────────
# 8. R² Heatmap — Models × Targets
# ─────────────────────────────────────────────────────────────
print("\nSTEP 8 — R² Heatmap")

r2_pivot = test_metrics.pivot(index="Model", columns="Target", values="R2")
r2_pivot = r2_pivot.loc[models]   # preserve model order

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(r2_pivot, annot=True, fmt=".4f", cmap="RdYlGn",
            vmin=0.7, vmax=1.0, linewidths=0.5,
            annot_kws={"size": 11, "weight": "bold"}, ax=ax)
bold_ax(ax, "R² Score Heatmap — All Models × All Targets", "Target", "Model")
plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold")
plt.yticks(rotation=0, fontsize=10, fontweight="bold")
fig.tight_layout()
save_fig(fig, "r2_heatmap_models_targets")

# ─────────────────────────────────────────────────────────────
# 9. Violin Plots — Error Distribution
# ─────────────────────────────────────────────────────────────
print("\nSTEP 9 — Violin Plots")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, tgt in enumerate(TARGETS):
    ax   = axes[i]
    meta = TARGET_META[tgt]

    vdata = []
    vlabs = []
    for m in models:
        if m not in preds:
            continue
        df_pred   = preds[m]
        residuals = (df_pred[f"{tgt}_actual_phys"].values
                     - df_pred[f"{tgt}_pred_phys"].values)
        vdata.append(residuals)
        vlabs.append(m)

    parts = ax.violinplot(vdata, showmedians=True,
                          showextrema=True)
    for j, pc in enumerate(parts["bodies"]):
        m = vlabs[j]
        pc.set_facecolor(MODEL_COLORS.get(m, "#aaa"))
        pc.set_alpha(0.75)

    ax.set_xticks(range(1, len(vlabs)+1))
    ax.set_xticklabels(vlabs, rotation=30, fontsize=9, fontweight="bold")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    bold_ax(ax,
            f"Error Distribution — {meta['symbol']}",
            "Model", "Residual (physical units)")

fig.suptitle("Prediction Error Violin Plots — All Models & Targets",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "violin_error_distribution")

# ─────────────────────────────────────────────────────────────
# 10. Model Ranking Summary
# ─────────────────────────────────────────────────────────────
print("\nSTEP 10 — Model Ranking Summary")

avg_metrics = (test_metrics.groupby("Model")
               [["MSE","RMSE","MAE","R2"]].mean()
               .sort_values("R2", ascending=False)
               .reset_index())
avg_metrics["Rank"] = range(1, len(avg_metrics)+1)

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
metric_labels = {
    "R2"  : "R² (↑ better)",
    "RMSE": "RMSE (↓ better)",
    "MAE" : "MAE (↓ better)",
    "MSE" : "MSE (↓ better)",
}

for ax, (metric, label) in zip(axes, metric_labels.items()):
    order_df = avg_metrics.sort_values(
        metric, ascending=(metric != "R2")
    )
    colors = [MODEL_COLORS.get(m, "#aaa") for m in order_df["Model"]]
    bars = ax.barh(order_df["Model"], order_df[metric],
                   color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, order_df[metric]):
        ax.text(bar.get_width() + bar.get_width()*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center",
                fontsize=9, fontweight="bold")
    bold_ax(ax, label, label, "Model")
    ax.invert_yaxis()

# Legend
patches = [mpatches.Patch(color=MODEL_COLORS.get(m, "#aaa"), label=m)
           for m in avg_metrics["Model"]]
fig.legend(handles=patches, loc="lower center", ncol=len(models),
           fontsize=10, framealpha=0.9, title="Models",
           title_fontsize=11)
fig.suptitle("Average Model Performance Ranking (Test Set)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0.08, 1, 1])
save_fig(fig, "model_ranking_summary")

# ─────────────────────────────────────────────────────────────
# 11. Per-Target Best Model Comparison
# ─────────────────────────────────────────────────────────────
print("\nSTEP 11 — Per-Target Best Model")

fig, ax = plt.subplots(figsize=(13, 7))
for tgt in TARGETS:
    subset = test_metrics[test_metrics["Target"]==tgt].sort_values("R2",ascending=False)
    x      = np.arange(len(subset))
    ax.plot(subset["Model"].values, subset["R2"].values,
            marker="o", linewidth=2, markersize=8,
            label=TARGET_META[tgt]["symbol"],
            color=TARGET_COLORS[tgt])
    for xi, (m, r2) in enumerate(zip(subset["Model"], subset["R2"])):
        ax.annotate(f"{r2:.3f}", (m, r2),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, fontweight="bold")

bold_ax(ax, "R² per Target Property — All Models",
        "Model", "R² Score")
ax.set_ylim(0.5, 1.05)
ax.axhline(0.9, color="gray", linestyle="--", linewidth=1)
ax.legend(fontsize=10)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
save_fig(fig, "r2_per_target_per_model")

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("05_evaluation_plots.py COMPLETE")
print(f"   All plots → {DIR_PLOTS}")
print("=" * 60)