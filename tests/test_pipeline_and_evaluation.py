"""Tests for the full pipeline and evaluation."""

import warnings

import numpy as np
import pandas as pd
import pytest

# Silence sklearn deprecation warnings unrelated to our code.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

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
from src.models import build_pipeline, make_classifier


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.fixture(scope="class")
    def small_data(self):
        df = generate_synthetic_athletes(
            SyntheticConfig(n_athletes=60, n_weeks=15, seed=0)
        )
        tr, te = athlete_holdout_split(df, test_frac=0.2, seed=0)
        return tr, te

    @pytest.mark.parametrize("kind", ["logreg", "random_forest", "gradient_boosting"])
    def test_pipeline_trains_and_predicts(self, small_data, kind):
        tr, te = small_data
        y_tr = tr["injured"].astype(int).to_numpy()
        pipeline = build_pipeline(model_kind=kind, random_state=0)
        pipeline.fit(tr, y_tr)
        proba = pipeline.predict_proba(te)[:, 1]
        assert proba.shape == (len(te),)
        assert np.all((0 <= proba) & (proba <= 1))

    def test_predict_returns_classes(self, small_data):
        tr, te = small_data
        y_tr = tr["injured"].astype(int).to_numpy()
        pipeline = build_pipeline(model_kind="logreg", random_state=0)
        pipeline.fit(tr, y_tr)
        preds = pipeline.predict(te)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_fitted_pipeline_handles_missing_optional_cols(self, small_data):
        tr, te = small_data
        y_tr = tr["injured"].astype(int).to_numpy()
        pipeline = build_pipeline(model_kind="logreg", random_state=0)
        pipeline.fit(tr, y_tr)

        # Drop an optional column from the test set.
        te_no_rpe = te.drop(columns=["rpe_avg"])
        # The FeatureEngineer pads it to NaN, the imputer fills it.
        proba = pipeline.predict_proba(te_no_rpe)[:, 1]
        assert proba.shape == (len(te_no_rpe),)

    def test_pipeline_serialization(self, small_data, tmp_path):
        import joblib

        tr, te = small_data
        y_tr = tr["injured"].astype(int).to_numpy()
        pipeline = build_pipeline(model_kind="random_forest", random_state=0)
        pipeline.fit(tr, y_tr)
        proba_orig = pipeline.predict_proba(te)[:, 1]

        path = tmp_path / "p.joblib"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)
        proba_loaded = loaded.predict_proba(te)[:, 1]
        np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-10)


# ---------------------------------------------------------------------------
# make_classifier
# ---------------------------------------------------------------------------


class TestMakeClassifier:
    def test_logreg(self):
        clf = make_classifier("logreg")
        assert clf.__class__.__name__ == "LogisticRegression"

    def test_random_forest(self):
        clf = make_classifier("random_forest")
        assert clf.__class__.__name__ == "RandomForestClassifier"

    def test_gradient_boosting(self):
        clf = make_classifier("gradient_boosting")
        assert clf.__class__.__name__ == "GradientBoostingClassifier"

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            make_classifier("not-a-real-model")

    def test_kwargs_override(self):
        clf = make_classifier("random_forest", n_estimators=50)
        assert clf.n_estimators == 50


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        # Probabilities that perfectly separate.
        p = np.array([0.1, 0.2, 0.8, 0.9])
        rep = evaluate(y, p, threshold=0.5)
        assert rep.roc_auc == 1.0
        assert rep.f1 == 1.0
        assert rep.precision == 1.0
        assert rep.recall == 1.0

    def test_no_signal(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200)
        p = rng.uniform(size=200)
        rep = evaluate(y, p)
        # ROC-AUC should be near 0.5 for random predictions.
        assert abs(rep.roc_auc - 0.5) < 0.1

    def test_metric_ranges(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=100)
        p = rng.uniform(size=100)
        rep = evaluate(y, p)
        assert 0.0 <= rep.roc_auc <= 1.0
        assert 0.0 <= rep.pr_auc <= 1.0
        assert 0.0 <= rep.f1 <= 1.0
        assert 0.0 <= rep.brier <= 1.0

    def test_confusion_matrix(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.6, 0.7, 0.4, 0.9])  # threshold 0.5: pred = [0, 1, 1, 0, 1]
        rep = evaluate(y, p, threshold=0.5)
        # TN=1, FP=1, FN=1, TP=2
        assert rep.confusion == [[1, 1], [1, 2]]

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            evaluate(np.array([0, 1]), np.array([0.5]))

    def test_non_binary_rejected(self):
        with pytest.raises(ValueError):
            evaluate(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

    def test_curves_included(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        rep = evaluate(y, p, include_curves=True)
        assert rep.threshold_curve is not None
        assert "roc_fpr" in rep.threshold_curve
        assert "pr_precision" in rep.threshold_curve


class TestBestF1Threshold:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        t, f1 = best_f1_threshold(y, p)
        assert f1 == 1.0

    def test_returns_in_unit_interval(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        p = rng.uniform(size=100)
        t, f1 = best_f1_threshold(y, p)
        assert 0.0 <= t <= 1.0


class TestGroupCrossValidate:
    def test_runs_and_returns_per_fold_metrics(self):
        df = generate_synthetic_athletes(
            SyntheticConfig(n_athletes=40, n_weeks=10, seed=0)
        )
        y = df["injured"].astype(int).to_numpy()
        groups = df["athlete_id"].to_numpy()
        pipeline = build_pipeline(model_kind="logreg", random_state=0)

        results = group_cross_validate(pipeline, df, y, groups=groups,
                                        n_splits=3)
        assert "roc_auc" in results
        assert len(results["roc_auc"]) == 3
        assert "pr_auc" in results

    def test_summarize_returns_mean_std(self):
        results = {
            "roc_auc": [0.7, 0.8, 0.6, 0.75],
            "f1": [0.3, 0.4, 0.35, 0.45],
        }
        summary = summarize_cv(results)
        assert summary["roc_auc"]["mean"] == pytest.approx(0.7125)
        assert summary["f1"]["mean"] == pytest.approx(0.375)
        assert summary["f1"]["min"] == 0.3
        assert summary["f1"]["max"] == 0.45
