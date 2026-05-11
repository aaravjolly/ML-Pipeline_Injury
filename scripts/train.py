"""
Train an injury-risk classifier end-to-end.

Workflow
--------
1. Load synthetic data (or a CSV via --data).
2. Hold out a fraction of *athletes* (not rows) for the test set.
3. Run grouped K-fold cross-validation on the training athletes to
   estimate generalization error.
4. Refit on the full training set, tune the decision threshold by F1
   on the validation fold of the last CV split, and evaluate on the
   held-out test set.
5. Save the entire pipeline + metadata to models/pipeline.joblib.
6. Render diagnostic figures (ROC, PR, calibration, confusion,
   feature importance).

Examples
--------
    python scripts/train.py
    python scripts/train.py --model logreg
    python scripts/train.py --data data/athletes.csv --model gradient_boosting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    SyntheticConfig,
    athlete_holdout_split,
    generate_synthetic_athletes,
    load_csv,
)
from src.evaluation import (
    best_f1_threshold,
    evaluate,
    group_cross_validate,
    summarize_cv,
)
from src.features import FeatureEngineer
from src.models import build_pipeline
from src.viz import (
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=None,
                   help="CSV path. If omitted, synthetic data is generated.")
    p.add_argument("--model", choices=["logreg", "random_forest", "gradient_boosting"],
                   default="gradient_boosting")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Synthetic data knobs (ignored if --data is given).
    p.add_argument("--n-athletes", type=int, default=200)
    p.add_argument("--n-weeks", type=int, default=26)

    # Output paths.
    p.add_argument("--out", type=Path, default=ROOT / "models" / "pipeline.joblib")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.data is not None:
        print(f"[load] reading {args.data}")
        df = load_csv(args.data)
    else:
        print(f"[load] generating synthetic data: "
              f"n_athletes={args.n_athletes} n_weeks={args.n_weeks}")
        df = generate_synthetic_athletes(SyntheticConfig(
            n_athletes=args.n_athletes,
            n_weeks=args.n_weeks,
            seed=args.seed,
        ))
    print(f"[load] {len(df)} records, {df['athlete_id'].nunique()} athletes, "
          f"injury rate {df['injured'].mean():.3f}")

    # ------------------------------------------------------------------
    # 2. Athlete-level train/test split
    # ------------------------------------------------------------------
    df_tr, df_te = athlete_holdout_split(df, test_frac=args.test_frac, seed=args.seed)
    print(f"[split] train={len(df_tr)} ({df_tr['athlete_id'].nunique()} athletes), "
          f"test={len(df_te)} ({df_te['athlete_id'].nunique()} athletes)")

    y_tr = df_tr["injured"].astype(int).to_numpy()
    y_te = df_te["injured"].astype(int).to_numpy()
    groups_tr = df_tr["athlete_id"].to_numpy()

    # ------------------------------------------------------------------
    # 3. Build pipeline
    # ------------------------------------------------------------------
    print(f"[build] model={args.model}")
    pipeline = build_pipeline(model_kind=args.model, random_state=args.seed)

    # ------------------------------------------------------------------
    # 4. Grouped cross-validation on training athletes
    # ------------------------------------------------------------------
    print(f"[cv] running {args.cv_folds}-fold grouped CV by athlete_id")
    t0 = time.perf_counter()
    cv_results = group_cross_validate(
        pipeline, df_tr, y_tr, groups=groups_tr,
        n_splits=args.cv_folds, threshold=0.5,
    )
    cv_summary = summarize_cv(cv_results)
    print(f"[cv] complete in {time.perf_counter() - t0:.1f}s")
    print()
    print("[cv] grouped K-fold results (mean ± std)")
    for metric, stats in cv_summary.items():
        print(f"     {metric:>10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # ------------------------------------------------------------------
    # 5. Refit on full training set + threshold tuning
    # ------------------------------------------------------------------
    print("\n[fit] refitting on full training set")
    pipeline.fit(df_tr, y_tr)

    # Tune threshold using a single held-out CV fold (to avoid optimizing
    # on the test set).
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=args.cv_folds)
    splits = list(cv.split(df_tr, y_tr, groups=groups_tr))
    tr_idx, va_idx = splits[-1]
    from sklearn.base import clone
    holdout_pipeline = clone(pipeline)
    holdout_pipeline.fit(df_tr.iloc[tr_idx], y_tr[tr_idx])
    proba_va = holdout_pipeline.predict_proba(df_tr.iloc[va_idx])[:, 1]
    best_thresh, best_f1 = best_f1_threshold(y_tr[va_idx], proba_va)
    print(f"[tune] best F1 threshold on internal validation: "
          f"{best_thresh:.3f} (F1 = {best_f1:.4f})")

    # ------------------------------------------------------------------
    # 6. Evaluate on the held-out test athletes
    # ------------------------------------------------------------------
    proba_te = pipeline.predict_proba(df_te)[:, 1]
    test_report = evaluate(y_te, proba_te, threshold=best_thresh, include_curves=False)

    print("\n[test] evaluation on held-out athletes")
    rep = test_report.as_dict()
    print(f"     samples = {rep['n_samples']}  positives = {rep['n_positives']} "
          f"({rep['base_rate']:.3f})")
    print(f"     ROC-AUC = {rep['roc_auc']:.4f}")
    print(f"     PR-AUC  = {rep['pr_auc']:.4f}")
    print(f"     F1      = {rep['f1']:.4f}  @threshold = {rep['threshold']:.3f}")
    print(f"     prec    = {rep['precision']:.4f}  recall = {rep['recall']:.4f}")
    print(f"     Brier   = {rep['brier']:.4f}")
    print(f"     confusion = TN={rep['confusion'][0][0]} FP={rep['confusion'][0][1]} "
          f"FN={rep['confusion'][1][0]} TP={rep['confusion'][1][1]}")

    # ------------------------------------------------------------------
    # 7. Persist model + metadata
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_kind": args.model,
        "best_threshold": best_thresh,
        "cv_summary": cv_summary,
        "test_metrics": test_report.as_dict(),
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "n_train_athletes": int(df_tr["athlete_id"].nunique()),
        "n_test_athletes": int(df_te["athlete_id"].nunique()),
        "feature_names": pipeline.named_steps["features"].feature_names_,
    }
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, args.out)

    # JSON sidecar so the README and CI can parse summary stats.
    sidecar = args.out.with_suffix(".joblib.json")
    sidecar.write_text(json.dumps({
        "model_kind": args.model,
        "best_threshold": best_thresh,
        "cv_summary": cv_summary,
        "test_metrics": test_report.as_dict(),
    }, indent=2))
    print(f"\n[save] pipeline saved to {args.out}")
    print(f"[save] metrics summary at {sidecar}")

    # ------------------------------------------------------------------
    # 8. Diagnostic figures
    # ------------------------------------------------------------------
    if not args.no_figures:
        try:
            args.fig_dir.mkdir(parents=True, exist_ok=True)
            plot_roc(y_te, proba_te, out_path=args.fig_dir / "roc.png")
            plot_pr(y_te, proba_te, out_path=args.fig_dir / "pr.png")
            plot_calibration(y_te, proba_te, out_path=args.fig_dir / "calibration.png")
            y_pred = (proba_te >= best_thresh).astype(int)
            plot_confusion_matrix(y_te, y_pred, out_path=args.fig_dir / "confusion.png")
            plot_feature_importance(pipeline, top_n=15,
                                     out_path=args.fig_dir / "feature_importance.png")
            print(f"[viz] figures saved to {args.fig_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] WARN: figure generation failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
