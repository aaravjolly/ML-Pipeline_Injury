"""
Model factory and end-to-end pipeline.

The full pipeline is::

    raw athlete-week DataFrame
        |
        v
    FeatureEngineer    (custom transformer; produces engineered features)
        |
        v
    ColumnTransformer  (numeric: median impute + StandardScaler;
                        categorical: most-frequent impute + one-hot)
        |
        v
    Classifier         (logistic / random forest / gradient boosting)

Wrapping everything in one ``sklearn.Pipeline`` means:
- Fit/transform discipline is enforced by sklearn (no test-set leakage)
- Cross-validation works naturally (every CV fold rebuilds the whole stack)
- Persistence is one ``joblib.dump`` for the entire artifact
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureEngineer, numeric_and_categorical_columns


ModelKind = Literal["logreg", "random_forest", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def make_classifier(kind: ModelKind, random_state: int = 0, **kwargs):
    """Construct a classifier by name with sensible defaults."""
    kind = kind.lower()
    if kind == "logreg":
        defaults = dict(
            C=1.0, class_weight="balanced",
            max_iter=2000, solver="lbfgs",
        )
        defaults.update(kwargs)
        return LogisticRegression(random_state=random_state, **defaults)

    if kind == "random_forest":
        defaults = dict(
            n_estimators=300, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1,
        )
        defaults.update(kwargs)
        return RandomForestClassifier(random_state=random_state, **defaults)

    if kind == "gradient_boosting":
        defaults = dict(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8,
        )
        defaults.update(kwargs)
        return GradientBoostingClassifier(random_state=random_state, **defaults)

    raise ValueError(
        f"Unknown model kind {kind!r}. "
        "Use 'logreg', 'random_forest', or 'gradient_boosting'."
    )


# ---------------------------------------------------------------------------
# Preprocessing layer (numeric / categorical split via ColumnTransformer)
# ---------------------------------------------------------------------------


def _build_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes + scales numerics and one-hot
    encodes categoricals.

    The feature_names argument is the *output* of the feature engineer;
    we partition it into numeric and categorical here.
    """
    parts = numeric_and_categorical_columns(feature_names)
    transformers = []
    if parts["numeric"]:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            parts["numeric"],
        ))
    if parts["categorical"]:
        # sklearn 1.2+ uses sparse_output. Fall back gracefully on older
        # versions where the kw was named ``sparse``.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore[arg-type]
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]),
            parts["categorical"],
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def build_pipeline(
    model_kind: ModelKind = "gradient_boosting",
    random_state: int = 0,
    feature_engineer: Optional[FeatureEngineer] = None,
    model_kwargs: Optional[Dict] = None,
) -> Pipeline:
    """
    Build the full FeatureEngineer -> Preprocessor -> Classifier pipeline.

    The preprocessor is dynamically constructed once we know the feature
    engineer's output columns, which we discover by fitting it on a tiny
    placeholder DataFrame at pipeline-build time. (sklearn pipelines need
    fixed transformer specs at construction.)

    Note: the actual numeric/categorical column lists are baked in here
    based on what the FeatureEngineer's default config outputs. If you
    customize the engineer (e.g., adding new categorical columns), pass
    a fitted ``feature_engineer`` so we can read the column list from it.
    """
    fe = feature_engineer or FeatureEngineer()
    if fe.feature_names_ is None:
        # Build a dummy frame so we can fit the engineer and discover columns.
        dummy = _make_dummy_frame()
        fe.fit(dummy)

    preprocessor = _build_preprocessor(fe.feature_names_)
    clf = make_classifier(model_kind, random_state=random_state, **(model_kwargs or {}))

    return Pipeline([
        ("features", fe),
        ("preprocess", preprocessor),
        ("classifier", clf),
    ])


def _make_dummy_frame() -> pd.DataFrame:
    """Tiny placeholder used to fit the FeatureEngineer at build time."""
    return pd.DataFrame({
        "athlete_id": ["A0", "A0", "A0", "A0", "A1", "A1"],
        "week": [0, 1, 2, 3, 0, 1],
        "age": [22.0, 22.0, 22.0, 22.0, 28.0, 28.0],
        "sex": ["M", "M", "M", "M", "F", "F"],
        "position": ["forward"] * 4 + ["defender"] * 2,
        "weekly_load": [1500.0, 1600.0, 1400.0, 1800.0, 1200.0, 1300.0],
        "sleep_hours": [7.0, 6.5, 8.0, 7.0, 7.5, 8.0],
        "soreness": [4, 5, 3, 6, 3, 4],
        "rpe_avg": [6.0, 7.0, 5.5, 7.0, 6.0, 6.5],
        "sessions_count": [5, 5, 4, 6, 4, 5],
        "sprint_distance_km": [3.0, 3.5, 2.5, 4.0, 2.0, 2.5],
        "prior_injuries": [0, 0, 0, 0, 1, 1],
        "injured": [0, 0, 0, 1, 0, 0],
    })
