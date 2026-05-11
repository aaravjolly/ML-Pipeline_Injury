"""
Pytest-free pipeline validator.

Runs the full pipeline end-to-end on a small synthetic dataset:
data generation -> feature engineering -> training -> CV ->
evaluation -> persistence -> reload -> prediction.

If anything is wrong with the pipeline glue (column mismatches, leakage,
serialization bugs), this catches it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    SyntheticConfig,
    athlete_holdout_split,
    generate_synthetic_athletes,
)
from src.evaluation import (
    best_f1_threshold,
    evaluate,
    group_cross_validate,
    summarize_cv,
)
from src.features import FeatureEngineer
from src.models import build_pipeline


def section(name):
    print(f"\n=== {name} " + "=" * (60 - len(name)))


def main() -> int:
    fail = []

    # ------------------------------------------------------------------
    section("synthetic data generation")
    df = generate_synthetic_athletes(SyntheticConfig(
        n_athletes=80, n_weeks=20, seed=0
    ))
    print(f"  rows={len(df)}  athletes={df['athlete_id'].nunique()}  "
          f"injury rate={df['injured'].mean():.3f}")
    # Required columns present.
    for col in ["athlete_id", "week", "weekly_load", "sleep_hours",
                "soreness", "injured"]:
        assert col in df.columns, f"missing required column: {col}"
    # Some missing values on optional columns (we deliberately injected ~8%).
    has_missing = df[["rpe_avg", "sessions_count", "sprint_distance_km"]].isna().any().any()
    assert has_missing, "expected some NaNs in optional columns"
    # Class imbalance check: this should be a minority class.
    assert 0.01 < df["injured"].mean() < 0.5
    print("  data generation OK")

    # Reproducibility.
    df2 = generate_synthetic_athletes(SyntheticConfig(
        n_athletes=80, n_weeks=20, seed=0
    ))
    assert df.equals(df2), "synthetic generator not deterministic"
    print("  deterministic with same seed OK")

    # ------------------------------------------------------------------
    section("athlete-level split has no overlapping athletes")
    df_tr, df_te = athlete_holdout_split(df, test_frac=0.2, seed=0)
    train_ids = set(df_tr["athlete_id"].unique())
    test_ids = set(df_te["athlete_id"].unique())
    assert train_ids.isdisjoint(test_ids), "athletes leak across split!"
    print(f"  train athletes={len(train_ids)}  test athletes={len(test_ids)} OK")

    # ------------------------------------------------------------------
    section("feature engineer produces engineered columns")
    fe = FeatureEngineer()
    Xt = fe.fit_transform(df_tr.copy())
    # The label must be removed from the output.
    assert "injured" not in Xt.columns, "label leaked into features!"
    assert "athlete_id" not in Xt.columns, "id should be dropped"
    # Critical engineered features should be present.
    for col in ["acwr", "load_change_pct", "sleep_4w_mean", "sleep_debt",
                "soreness_3w_max", "injury_history_4w"]:
        assert col in Xt.columns, f"missing engineered feature: {col}"
    print(f"  output shape={Xt.shape}  has_features={list(Xt.columns)[:5]}...")

    # ACWR for week 0 of each athlete should equal 1.0 (load / load).
    df_sample = df_tr.copy().sort_values(["athlete_id", "week"]).reset_index(drop=True)
    Xt_sample = fe.transform(df_sample)
    week0 = df_sample.groupby("athlete_id").head(1).index
    acwr_week0 = Xt_sample.loc[week0, "acwr"].values
    assert np.allclose(acwr_week0, 1.0, atol=1e-6), \
        f"ACWR at week 0 should be 1.0, got {acwr_week0[:5]}"
    print(f"  ACWR at week 0 == 1.0 for all athletes OK")

    # injury_history_4w must NOT depend on the current week's label.
    # Verify by flipping the current week's label and checking the feature
    # is unchanged.
    flipped = df_sample.copy()
    flipped["injured"] = 1 - flipped["injured"]
    Xt_flipped = fe.transform(flipped)
    # The injury_history feature should be identical because it shifts by 1.
    # But changing every label changes the *prior* week label too, which
    # WILL change the feature. So flip just one row instead.
    only_one = df_sample.copy()
    target_idx = 100
    only_one.loc[target_idx, "injured"] = 1 - only_one.loc[target_idx, "injured"]
    Xt_one = fe.transform(only_one)
    # Check that the row at target_idx itself has the same injury_history_4w
    # as the original (since the feature uses only PRIOR weeks).
    orig_val = Xt_sample.loc[target_idx, "injury_history_4w"]
    new_val = Xt_one.loc[target_idx, "injury_history_4w"]
    assert orig_val == new_val, \
        f"injury_history_4w leaks current-week label: {orig_val} vs {new_val}"
    print("  injury_history_4w does NOT use current-week label OK")

    # ------------------------------------------------------------------
    section("end-to-end pipeline trains and predicts")
    pipeline = build_pipeline(model_kind="gradient_boosting", random_state=0)
    y_tr = df_tr["injured"].astype(int).to_numpy()

    pipeline.fit(df_tr, y_tr)
    proba = pipeline.predict_proba(df_te)[:, 1]
    assert proba.shape == (len(df_te),)
    assert np.all((0 <= proba) & (proba <= 1))
    print(f"  trained, predicted on {len(df_te)} rows; "
          f"proba range=[{proba.min():.3f}, {proba.max():.3f}] OK")

    # ------------------------------------------------------------------
    section("evaluation report")
    y_te = df_te["injured"].astype(int).to_numpy()
    rep = evaluate(y_te, proba, threshold=0.5)
    print(f"  ROC-AUC={rep.roc_auc:.4f}  PR-AUC={rep.pr_auc:.4f}  "
          f"F1={rep.f1:.4f}")
    print(f"  precision={rep.precision:.4f}  recall={rep.recall:.4f}  "
          f"brier={rep.brier:.4f}")
    if rep.roc_auc < 0.55:
        fail.append(f"ROC-AUC unexpectedly low ({rep.roc_auc}); pipeline may be broken")
    else:
        print("  model is meaningfully better than chance OK")

    # ------------------------------------------------------------------
    section("threshold tuning")
    best_t, best_f1 = best_f1_threshold(y_te, proba)
    print(f"  best F1 threshold = {best_t:.3f}  (F1 = {best_f1:.4f})")
    assert 0.0 <= best_t <= 1.0
    assert best_f1 >= rep.f1 - 1e-6, "tuned F1 should be >= F1 at default threshold"
    print("  tuned F1 >= default-threshold F1 OK")

    # ------------------------------------------------------------------
    section("grouped CV (no athlete leakage between folds)")
    cv_pipeline = build_pipeline(model_kind="logreg", random_state=0)
    cv_results = group_cross_validate(
        cv_pipeline, df_tr, y_tr,
        groups=df_tr["athlete_id"].to_numpy(),
        n_splits=5, threshold=0.5,
    )
    summary = summarize_cv(cv_results)
    print(f"  ROC-AUC: {summary['roc_auc']['mean']:.4f} ± "
          f"{summary['roc_auc']['std']:.4f} "
          f"(min={summary['roc_auc']['min']:.4f}, max={summary['roc_auc']['max']:.4f})")
    print(f"  PR-AUC : {summary['pr_auc']['mean']:.4f} ± "
          f"{summary['pr_auc']['std']:.4f}")
    if summary["roc_auc"]["mean"] < 0.55:
        fail.append(f"CV ROC-AUC too low: {summary['roc_auc']['mean']}")
    else:
        print("  CV results show meaningful signal OK")

    # ------------------------------------------------------------------
    section("save / load roundtrip")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pipeline.joblib"
        joblib.dump({"pipeline": pipeline, "metadata": {"x": 1}}, path)
        loaded = joblib.load(path)
        proba_reload = loaded["pipeline"].predict_proba(df_te)[:, 1]
        assert np.allclose(proba_reload, proba, atol=1e-10), \
            "reloaded pipeline produces different predictions"
    print("  reloaded pipeline matches original predictions OK")

    # ------------------------------------------------------------------
    section("multiple model kinds train cleanly")
    for kind in ["logreg", "random_forest", "gradient_boosting"]:
        p = build_pipeline(model_kind=kind, random_state=0)
        p.fit(df_tr, y_tr)
        prb = p.predict_proba(df_te)[:, 1]
        rep_k = evaluate(y_te, prb)
        print(f"  {kind:18s}  ROC-AUC={rep_k.roc_auc:.4f}  "
              f"F1={rep_k.f1:.4f}  brier={rep_k.brier:.4f}")
        if not (0 <= prb.min() and prb.max() <= 1):
            fail.append(f"{kind} produced out-of-range probabilities")

    # ------------------------------------------------------------------
    section("error handling")
    # Non-binary labels.
    try:
        bad_y = np.array([0, 1, 2, 1, 0])
        evaluate(bad_y, np.array([0.1, 0.5, 0.7, 0.4, 0.2]))
    except ValueError:
        print("  non-binary labels rejected OK")
    else:
        fail.append("expected ValueError on non-binary labels")

    # Shape mismatch.
    try:
        evaluate(np.array([0, 1, 0]), np.array([0.1, 0.5]))
    except ValueError:
        print("  shape mismatch rejected OK")
    else:
        fail.append("expected ValueError on shape mismatch")

    # FeatureEngineer missing required columns.
    try:
        bad_df = df_tr.drop(columns=["weekly_load"])
        FeatureEngineer().fit(bad_df)
    except ValueError:
        print("  missing required column rejected OK")
    else:
        fail.append("expected ValueError on missing required column")

    # Unknown model kind.
    try:
        from src.models import make_classifier
        make_classifier("not-a-real-model")
    except ValueError:
        print("  unknown model kind rejected OK")
    else:
        fail.append("expected ValueError on unknown model kind")

    # ------------------------------------------------------------------
    section("summary")
    if fail:
        print("FAILURES:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
