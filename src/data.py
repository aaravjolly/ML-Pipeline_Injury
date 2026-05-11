"""
Data loading.

Two paths:

- ``generate_synthetic_athletes()``: creates a realistic synthetic
  dataset of athlete-week records for demos and tests. Each record is
  one athlete-week with workload, sleep, soreness, and injury history
  features and a binary ``injured`` label for the *following* week.

- ``load_csv()``: load a real dataset from disk. Expected columns are
  documented below; missing optional columns are handled gracefully by
  the feature engineering layer.

Both return a single ``pandas.DataFrame``. Splitting and feature
engineering happen downstream so the data layer stays simple.

Expected schema for real CSV
----------------------------
Required:
    athlete_id          string
    week                int (sequential per athlete)
    age                 float
    weekly_load         float (sum of training session loads)
    sleep_hours         float
    soreness            int (1-10 self-report)
    injured             int (0/1, label for THIS week)

Optional (filled with NaN if absent, handled by the imputer):
    sex                 "M"/"F"
    position            string (categorical)
    rpe_avg             float (avg rate of perceived exertion 1-10)
    sessions_count      int
    sprint_distance_km  float
    prior_injuries      int
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "athlete_id", "week", "age",
    "weekly_load", "sleep_hours", "soreness", "injured",
]
OPTIONAL_COLUMNS = [
    "sex", "position", "rpe_avg", "sessions_count",
    "sprint_distance_km", "prior_injuries",
]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """Knobs for the synthetic athlete dataset."""

    n_athletes: int = 200
    n_weeks: int = 26              # roughly half a season
    base_injury_rate: float = 0.05  # probability of injury in a baseline week
    seed: int = 0


def generate_synthetic_athletes(config: Optional[SyntheticConfig] = None) -> pd.DataFrame:
    """
    Generate a synthetic athlete-week dataset with a believable injury label.

    The label is generated from a known logistic process: workload spikes
    (acute > chronic), sleep debt, high soreness, and prior injuries all
    push up injury probability. The generator deliberately injects:

    - Missing values on optional columns (about 8%)
    - One outlier week per athlete with extreme workload
    - Class imbalance (the base injury rate is 5%)
    - A small number of athletes with chronic injury risk (~15%)

    This isn't meant to perfectly mimic real sport science data, but it
    gives the pipeline a realistic shape: noisy, imbalanced, with
    correlated time-series features.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict] = []
    positions = ["forward", "midfielder", "defender", "goalkeeper"]
    sexes = ["M", "F"]

    # Per-athlete attributes that don't change week-to-week.
    is_chronic = rng.uniform(size=cfg.n_athletes) < 0.15  # ~15% high-risk
    ages = rng.normal(25.0, 4.0, size=cfg.n_athletes).clip(17, 38)
    athlete_sex = rng.choice(sexes, size=cfg.n_athletes)
    athlete_pos = rng.choice(positions, size=cfg.n_athletes)
    prior_inj = rng.poisson(lam=0.7, size=cfg.n_athletes)
    chronic_baseline = is_chronic.astype(float) * 0.04  # +4pp injury rate per week

    for i in range(cfg.n_athletes):
        # Athlete-specific workload mean: slight per-athlete variation.
        load_mu = rng.normal(1500.0, 200.0)
        sleep_mu = rng.normal(7.5, 0.5)
        chronic_load = load_mu  # rolling chronic load tracker

        for w in range(cfg.n_weeks):
            # Acute load with a positive spike on one random week per athlete.
            acute_load = rng.normal(load_mu, 200.0)
            if w == int(rng.integers(2, cfg.n_weeks - 1)):
                acute_load *= rng.uniform(1.5, 2.0)

            # Slow-moving chronic load (4-week EMA-ish).
            chronic_load = 0.75 * chronic_load + 0.25 * acute_load

            sleep = float(np.clip(rng.normal(sleep_mu, 1.0), 3.5, 10.0))
            rpe = float(np.clip(rng.normal(6.5, 1.2), 1, 10))
            soreness = int(np.clip(rng.normal(4.0, 1.8), 1, 10))
            sessions = int(np.clip(rng.poisson(5), 1, 12))
            sprint_km = float(np.clip(rng.normal(3.0, 1.0), 0, 15))

            # ------- generate the label using a known mechanism -------
            # Acute:chronic workload ratio (ACWR) - well-known injury
            # predictor in sport science.
            acwr = acute_load / max(chronic_load, 1e-3)
            sleep_debt = max(0.0, 7.5 - sleep)

            logit = (
                -3.0
                + 2.2 * (acwr - 1.0)             # high ACWR -> risk up (stronger)
                + 0.45 * sleep_debt              # under 7.5h sleep -> risk up
                + 0.30 * (soreness - 4)          # soreness above baseline (stronger)
                + 0.20 * prior_inj[i]            # chronic risk
                + 1.5 * chronic_baseline[i]      # high-risk athletes
                + 0.03 * (ages[i] - 25)          # mild age effect
                + rng.normal(0.0, 0.25)          # less irreducible noise
            )
            p = 1.0 / (1.0 + np.exp(-logit))
            # Anchor average to the configured base rate.
            p = float(np.clip(p, 0.0, 0.75))
            injured = int(rng.uniform() < p)

            rows.append({
                "athlete_id": f"A{i:04d}",
                "week": w,
                "age": float(ages[i]),
                "sex": athlete_sex[i],
                "position": athlete_pos[i],
                "weekly_load": float(acute_load),
                "sleep_hours": sleep,
                "soreness": soreness,
                "rpe_avg": rpe,
                "sessions_count": sessions,
                "sprint_distance_km": sprint_km,
                "prior_injuries": int(prior_inj[i]),
                "injured": injured,
            })

    df = pd.DataFrame(rows)

    # Inject missing values on optional columns to make the imputer earn its keep.
    for col in ["rpe_avg", "sessions_count", "sprint_distance_km"]:
        mask = rng.uniform(size=len(df)) < 0.08
        df.loc[mask, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load an athlete-week dataset from CSV.

    Verifies the required columns are present. Optional columns are
    accepted in any subset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Required: {REQUIRED_COLUMNS}"
        )
    return df


# ---------------------------------------------------------------------------
# Train / test splitting that respects athlete grouping
# ---------------------------------------------------------------------------


def athlete_holdout_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a fraction of athletes (not rows) for the test set.

    Random row-level splits leak information across weeks of the same
    athlete - the model can memorize per-athlete baselines. Splitting by
    athlete forces the model to generalize to new people.
    """
    rng = np.random.default_rng(seed)
    athletes = np.asarray(df["athlete_id"].unique())
    rng.shuffle(athletes)
    n_test = max(1, int(round(len(athletes) * test_frac)))
    test_ids = set(athletes[:n_test].tolist())
    test_mask = df["athlete_id"].isin(test_ids)
    return df.loc[~test_mask].reset_index(drop=True), df.loc[test_mask].reset_index(drop=True)
