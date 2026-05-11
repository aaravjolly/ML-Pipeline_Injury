"""End-to-end injury-risk ML pipeline."""

from .data import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SyntheticConfig,
    athlete_holdout_split,
    generate_synthetic_athletes,
    load_csv,
)
from .evaluation import (
    EvalReport,
    best_f1_threshold,
    evaluate,
    group_cross_validate,
    summarize_cv,
)
from .features import FeatureConfig, FeatureEngineer, split_xy
from .models import build_pipeline, make_classifier

__all__ = [
    # data
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SyntheticConfig",
    "athlete_holdout_split",
    "generate_synthetic_athletes",
    "load_csv",
    # features
    "FeatureConfig",
    "FeatureEngineer",
    "split_xy",
    # models
    "build_pipeline",
    "make_classifier",
    # evaluation
    "EvalReport",
    "best_f1_threshold",
    "evaluate",
    "group_cross_validate",
    "summarize_cv",
]

__version__ = "1.0.0"
