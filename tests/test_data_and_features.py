"""Tests for the data and feature-engineering modules."""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    SyntheticConfig,
    athlete_holdout_split,
    generate_synthetic_athletes,
    load_csv,
)
from src.features import FeatureConfig, FeatureEngineer


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


class TestSyntheticData:
    def test_basic_shape(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=20, n_weeks=10, seed=0))
        assert len(df) == 200
        assert df["athlete_id"].nunique() == 20
        # Required columns.
        for col in ["athlete_id", "week", "weekly_load", "sleep_hours",
                    "soreness", "injured"]:
            assert col in df.columns

    def test_label_is_binary(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=10, n_weeks=10, seed=0))
        assert set(df["injured"].unique()).issubset({0, 1})

    def test_class_imbalance(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=100, n_weeks=20, seed=0))
        rate = df["injured"].mean()
        # Should be a minority class but not zero.
        assert 0.01 < rate < 0.5

    def test_deterministic(self):
        a = generate_synthetic_athletes(SyntheticConfig(n_athletes=10, n_weeks=5, seed=42))
        b = generate_synthetic_athletes(SyntheticConfig(n_athletes=10, n_weeks=5, seed=42))
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self):
        a = generate_synthetic_athletes(SyntheticConfig(n_athletes=10, n_weeks=5, seed=0))
        b = generate_synthetic_athletes(SyntheticConfig(n_athletes=10, n_weeks=5, seed=999))
        # The labels should differ between seeds (with overwhelming probability).
        assert not a["injured"].equals(b["injured"])

    def test_optional_columns_have_nans(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=50, n_weeks=20, seed=0))
        # We deliberately inject ~8% missing values.
        assert df["rpe_avg"].isna().any()
        assert df["sessions_count"].isna().any()


# ---------------------------------------------------------------------------
# Athlete-level split
# ---------------------------------------------------------------------------


class TestAthleteSplit:
    def test_no_athlete_overlap(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=50, n_weeks=10, seed=0))
        tr, te = athlete_holdout_split(df, test_frac=0.2, seed=0)
        train_ids = set(tr["athlete_id"].unique())
        test_ids = set(te["athlete_id"].unique())
        assert train_ids.isdisjoint(test_ids)

    def test_split_sizes(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=100, n_weeks=10, seed=0))
        tr, te = athlete_holdout_split(df, test_frac=0.3, seed=0)
        assert tr["athlete_id"].nunique() + te["athlete_id"].nunique() == 100
        # ~30% of athletes in test.
        assert 25 <= te["athlete_id"].nunique() <= 35

    def test_reproducible(self):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=30, n_weeks=10, seed=0))
        a_tr, a_te = athlete_holdout_split(df, test_frac=0.2, seed=42)
        b_tr, b_te = athlete_holdout_split(df, test_frac=0.2, seed=42)
        assert a_tr.equals(b_tr)
        assert a_te.equals(b_te)


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_round_trip(self, tmp_path):
        df = generate_synthetic_athletes(SyntheticConfig(n_athletes=5, n_weeks=5, seed=0))
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        loaded = load_csv(p)
        assert len(loaded) == len(df)
        for col in ["athlete_id", "week", "weekly_load", "sleep_hours",
                    "soreness", "injured"]:
            assert col in loaded.columns

    def test_missing_required_raises(self, tmp_path):
        df = pd.DataFrame({"athlete_id": ["A0"], "week": [0]})  # missing required cols
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing"):
            load_csv(p)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class TestFeatureEngineer:
    @pytest.fixture
    def df(self):
        return generate_synthetic_athletes(
            SyntheticConfig(n_athletes=10, n_weeks=10, seed=0)
        )

    def test_basic_transform(self, df):
        fe = FeatureEngineer()
        out = fe.fit_transform(df)
        assert len(out) == len(df)
        assert "acwr" in out.columns
        assert "load_change_pct" in out.columns

    def test_label_dropped(self, df):
        fe = FeatureEngineer(drop_label=True)
        out = fe.fit_transform(df)
        assert "injured" not in out.columns

    def test_label_kept_when_requested(self, df):
        fe = FeatureEngineer(drop_label=False)
        out = fe.fit_transform(df)
        assert "injured" in out.columns

    def test_id_columns_dropped(self, df):
        fe = FeatureEngineer(drop_id_columns=True)
        out = fe.fit_transform(df)
        assert "athlete_id" not in out.columns
        assert "week" not in out.columns

    def test_acwr_at_week_zero_is_one(self, df):
        """For each athlete's week 0, ACWR = current_load / current_load = 1."""
        fe = FeatureEngineer(drop_id_columns=False)
        df_sorted = df.sort_values(["athlete_id", "week"]).reset_index(drop=True)
        out = fe.fit_transform(df_sorted)
        first_rows = df_sorted.groupby("athlete_id").head(1).index
        np.testing.assert_allclose(out.loc[first_rows, "acwr"].values, 1.0,
                                    atol=1e-6)

    def test_no_label_leakage_in_history_feature(self, df):
        """injury_history_4w should not depend on the *current* week's label."""
        fe = FeatureEngineer(drop_id_columns=False, drop_label=True)
        df_sorted = df.sort_values(["athlete_id", "week"]).reset_index(drop=True)
        out_orig = fe.fit_transform(df_sorted)
        # Flip one row's label.
        df_flipped = df_sorted.copy()
        target_idx = 50
        df_flipped.loc[target_idx, "injured"] = 1 - df_flipped.loc[target_idx, "injured"]
        out_flipped = fe.transform(df_flipped)
        # injury_history_4w at the *current* row should be identical
        # (it uses only prior weeks).
        assert (
            out_orig.loc[target_idx, "injury_history_4w"]
            == out_flipped.loc[target_idx, "injury_history_4w"]
        )

    def test_missing_required_columns_raises(self, df):
        fe = FeatureEngineer()
        bad = df.drop(columns=["weekly_load"])
        with pytest.raises(ValueError, match="weekly_load|missing"):
            fe.fit(bad)

    def test_get_feature_names_out(self, df):
        fe = FeatureEngineer()
        fe.fit(df)
        names = fe.get_feature_names_out()
        assert len(names) > 0
        assert "acwr" in names

    def test_handles_single_week_athletes(self):
        """An athlete with only one week of data should still produce features."""
        df = pd.DataFrame({
            "athlete_id": ["A0"],
            "week": [0],
            "age": [25.0],
            "weekly_load": [1500.0],
            "sleep_hours": [7.0],
            "soreness": [4],
            "injured": [0],
        })
        fe = FeatureEngineer()
        out = fe.fit_transform(df)
        assert len(out) == 1
        # ACWR should be 1.0 (only one week; chronic = current).
        assert abs(out["acwr"].iloc[0] - 1.0) < 1e-6
