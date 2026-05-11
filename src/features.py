"""
Feature engineering.

Builds derived features from the raw athlete-week records. The most
important ones are time-series features that have to be computed
*per athlete* and *up to but not including* the prediction week, to
avoid label leakage:

- ``acwr_4w``: acute:chronic workload ratio. The single most-cited
  predictor in the sport-science injury-risk literature. Computed as
  this week's load divided by the trailing 4-week mean.

- ``load_change_pct``: percent change in this week's load vs last week.

- ``sleep_4w_mean``: trailing-4-week sleep average.

- ``soreness_3w_max``: maximum self-reported soreness in the last 3
  weeks, including this one.

- ``injury_history_4w``: number of injured weeks in the prior 4 weeks
  (excluding this week so we don't leak the label).

The feature engineer is implemented as a sklearn ``BaseEstimator`` /
``TransformerMixin`` so it slots into a ``Pipeline`` and respects the
fit/transform contract. Statistics that depend on training data are
learned in ``fit`` (currently just per-athlete defaults for cold-start
predictions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Columns the engineer requires at fit/transform time.
REQUIRED_COLS = [
    "athlete_id", "week",
    "weekly_load", "sleep_hours", "soreness", "injured",
]


@dataclass
class FeatureConfig:
    """Tunables for the engineer."""

    chronic_window: int = 4
    history_window: int = 4
    soreness_window: int = 3


# ---------------------------------------------------------------------------
# Per-athlete rolling features
# ---------------------------------------------------------------------------


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (athlete_id, week) so groupby + shift behave deterministically."""
    return df.sort_values(["athlete_id", "week"], kind="stable").reset_index(drop=True)


def _add_rolling_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Group-by-athlete rolling features. Avoids future leakage."""
    df = df.copy()
    g = df.groupby("athlete_id", sort=False)

    # Chronic workload: trailing-N-week mean of weekly_load *including*
    # the current week. ACWR uses this as the denominator.
    df["chronic_load"] = (
        g["weekly_load"].transform(
            lambda s: s.rolling(window=cfg.chronic_window, min_periods=1).mean()
        )
    )
    df["acwr"] = df["weekly_load"] / df["chronic_load"].clip(lower=1e-3)

    # Week-over-week load change (in percent of previous week).
    prev_load = g["weekly_load"].shift(1)
    df["load_change_pct"] = (
        (df["weekly_load"] - prev_load) / prev_load.clip(lower=1e-3)
    ).fillna(0.0)

    # Trailing 4-week mean sleep (current week included).
    df[f"sleep_{cfg.chronic_window}w_mean"] = (
        g["sleep_hours"].transform(
            lambda s: s.rolling(cfg.chronic_window, min_periods=1).mean()
        )
    )
    df["sleep_debt"] = (7.5 - df[f"sleep_{cfg.chronic_window}w_mean"]).clip(lower=0.0)

    # Max soreness in the last few weeks.
    df[f"soreness_{cfg.soreness_window}w_max"] = (
        g["soreness"].transform(
            lambda s: s.rolling(cfg.soreness_window, min_periods=1).max()
        )
    )

    # Injury history: number of injured weeks in the prior N weeks
    # (NOT including this week - that would leak the label!).
    prior_injured = g["injured"].shift(1).fillna(0)
    df[f"injury_history_{cfg.history_window}w"] = (
        prior_injured.groupby(df["athlete_id"], sort=False)
        .transform(lambda s: s.rolling(cfg.history_window, min_periods=1).sum())
    )

    # Sessions per RPE - high training volume * high effort is risky.
    if "rpe_avg" in df.columns and "sessions_count" in df.columns:
        df["rpe_x_sessions"] = (
            df["rpe_avg"].fillna(df["rpe_avg"].median()) *
            df["sessions_count"].fillna(df["sessions_count"].median())
        )
    return df


# ---------------------------------------------------------------------------
# Sklearn-compatible feature engineer
# ---------------------------------------------------------------------------


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add engineered features to an athlete-week DataFrame.

    Parameters
    ----------
    config : FeatureConfig, optional
    drop_label : bool
        If True (default), drop the ``injured`` column from the output -
        it would leak the label into the model. Set False only if you
        explicitly need the label downstream (e.g., for evaluation).
    drop_id_columns : bool
        If True, drop ``athlete_id`` and ``week`` from the output. The
        downstream model shouldn't see them - they're for grouping in
        cross-validation.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        drop_label: bool = True,
        drop_id_columns: bool = True,
    ) -> None:
        self.config = config or FeatureConfig()
        self.drop_label = drop_label
        self.drop_id_columns = drop_id_columns
        # Set in ``fit``.
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        self._check_columns(X)
        # Run a transform on the training data once so we can capture
        # the resulting column list for downstream pipeline introspection.
        Xt = self._transform_impl(X)
        self.feature_names_ = list(Xt.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            # Sklearn calls transform on unfitted clones during clone() in
            # some CV paths; be tolerant.
            self.fit(X)
        self._check_columns(X)
        Xt = self._transform_impl(X)
        # Pad missing columns (e.g. optional cols not present at predict time).
        for col in self.feature_names_:
            if col not in Xt.columns:
                Xt[col] = np.nan
        # Reorder for stability.
        return Xt[self.feature_names_]

    def fit_transform(self, X: pd.DataFrame, y=None, **kw) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_sorted(X)
        df = _add_rolling_features(df, self.config)
        if self.drop_label and "injured" in df.columns:
            df = df.drop(columns=["injured"])
        if self.drop_id_columns:
            df = df.drop(columns=[c for c in ("athlete_id", "week") if c in df.columns])
        return df

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLS if c not in X.columns]
        if missing:
            raise ValueError(
                f"FeatureEngineer requires columns {missing} but they're missing. "
                f"Got: {list(X.columns)}"
            )

    # Sklearn 1.0+ feature-name introspection.
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.feature_names_ is None:
            raise RuntimeError("FeatureEngineer is not fitted")
        return np.asarray(self.feature_names_, dtype=object)


# ---------------------------------------------------------------------------
# Helpers used by the pipeline factory
# ---------------------------------------------------------------------------


def split_xy(df: pd.DataFrame, label_col: str = "injured"):
    """Return (X, y) where X keeps the raw frame for the FeatureEngineer."""
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not in DataFrame")
    y = df[label_col].astype(int).to_numpy()
    return df, y


def numeric_and_categorical_columns(
    feature_names: Sequence[str],
) -> Dict[str, List[str]]:
    """Partition the engineered feature names into numeric / categorical.

    The downstream ColumnTransformer needs to know which columns to
    standardize vs one-hot encode.
    """
    cat = [c for c in feature_names if c in {"sex", "position"}]
    num = [c for c in feature_names if c not in cat]
    return {"numeric": num, "categorical": cat}
