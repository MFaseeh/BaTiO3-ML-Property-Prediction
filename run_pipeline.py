# ============================================================
# run_pipeline.py — Master Pipeline Runner
# ============================================================
# Run this single script to execute the full pipeline in order.
# Or run individual steps independently.
#
# Usage:
#   python run_pipeline.py              # full pipeline
#   python run_pipeline.py --step 1    # single step
#   python run_pipeline.py --from 3    # from step 3 onward
# ============================================================

import subprocess
import sys
import argparse
import time
import os

STEPS = [
    (1, "01_preprocessing.py",    "Data Loading, Splitting & EDA"),
    (2, "02_feature_selection.py","RF + SHAP Consensus Feature Selection"),
    (3, "03_hpo.py",              "Bayesian HPO — All 7 Models"),
    (4, "04_training.py",         "Model Training + 5-Fold CV"),
    (5, "05_evaluation_plots.py", "Evaluation & Comparison Plots"),
    (6, "06_ablation.py",         "Ablation Studies"),
]

def run_step(num, script, desc):
    print("\n" + "=" * 65)
    print(f"  STEP {num}: {desc}")
    print(f"  Script : {script}")
    print("=" * 65)
    t0     = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ Step {num} FAILED after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"\n✅ Step {num} completed in {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="BaTiO3 ML Pipeline Runner")
    parser.add_argument("--step", type=int, default=None,
                        help="Run a single step (1-6)")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from step N (default=1)")
    args = parser.parse_args()

    print("\n" + "█" * 65)
    print("  BaTiO3 Multi-Property ML Pipeline")
    print("  Revised Pipeline — Reviewer Revisions Incorporated")
    print("█" * 65)

    # Verify dataset exists
    from config import DATA_FILE
    if not os.path.exists(DATA_FILE):
        print(f"\n❌ Dataset not found: {DATA_FILE}")
        print("   Place BA_expanded_zero_formula_coolrate_fixed.csv in this directory.")
        sys.exit(1)
    else:
        print(f"\n✅ Dataset found: {DATA_FILE}")

    t_start = time.time()

    if args.step:
        step_map = {s[0]: s for s in STEPS}
        if args.step not in step_map:
            print(f"❌ Invalid step: {args.step}. Choose 1–6.")
            sys.exit(1)
        num, script, desc = step_map[args.step]
        run_step(num, script, desc)
    else:
        for num, script, desc in STEPS:
            if num >= args.from_step:
                run_step(num, script, desc)

    total = time.time() - t_start
    print("\n" + "█" * 65)
    print(f"  ✅ PIPELINE COMPLETE  |  Total time: {total/60:.1f} min")
    print("█" * 65)
    print("\n  Output structure:")
    print("  outputs/")
    print("  ├── splits/          Train/val/test CSVs + indices")
    print("  ├── features/        Feature importance + variance CSVs")
    print("  ├── hpo/             Best hyperparameters JSON + summary")
    print("  ├── models/          Trained .joblib models")
    print("  ├── predictions/     Per-model prediction CSVs")
    print("  ├── metrics/         Test + CV + ablation metric CSVs")
    print("  ├── plots/           All PNG + PDF figures")
    print("  └── shap/            SHAP summary + dependence plots\n")

if __name__ == "__main__":
    main()
