"""
Command-line predictor.

Examples
--------
    # Predict for one athlete-week passed via flags.
    python scripts/predict.py --athlete-id A0001 --week 12 --age 24 \\
        --weekly-load 2200 --sleep 6.5 --soreness 6 --rpe 7.5

    # Predict for a CSV of athlete-weeks.
    python scripts/predict.py --csv data/new_athletes.csv --out predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path,
                   default=ROOT / "models" / "pipeline.joblib")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override decision threshold. Default: use the one "
                        "saved in the model bundle.")

    p.add_argument("--csv", type=Path, default=None,
                   help="CSV of records to predict for.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write predictions if --csv given.")

    # Single-record flags.
    p.add_argument("--athlete-id", default="UNKNOWN")
    p.add_argument("--week", type=int, default=0)
    p.add_argument("--age", type=float, default=25.0)
    p.add_argument("--weekly-load", type=float, default=1500.0)
    p.add_argument("--sleep", type=float, default=7.5)
    p.add_argument("--soreness", type=int, default=4)
    p.add_argument("--rpe", type=float, default=None)
    p.add_argument("--sessions", type=int, default=None)
    p.add_argument("--prior-injuries", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"[error] model not found: {args.model}", file=sys.stderr)
        return 1
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    metadata = bundle.get("metadata", {})
    threshold = args.threshold if args.threshold is not None else \
        metadata.get("best_threshold", 0.5)

    if args.csv is not None:
        df = pd.read_csv(args.csv)
        proba = pipeline.predict_proba(df)[:, 1]
        df_out = df.copy()
        df_out["risk_probability"] = proba
        df_out["predicted_injured"] = (proba >= threshold).astype(int)
        if args.out is not None:
            df_out.to_csv(args.out, index=False)
            print(f"[done] wrote {len(df_out)} predictions to {args.out}")
        else:
            print(df_out[["athlete_id", "week", "risk_probability",
                          "predicted_injured"]].to_string(index=False))
        return 0

    # Single record.
    record = {
        "athlete_id": args.athlete_id,
        "week": args.week,
        "age": args.age,
        "weekly_load": args.weekly_load,
        "sleep_hours": args.sleep,
        "soreness": args.soreness,
        "rpe_avg": args.rpe,
        "sessions_count": args.sessions,
        "prior_injuries": args.prior_injuries,
        "injured": 0,  # placeholder, dropped by FE
    }
    df = pd.DataFrame([record])
    proba = pipeline.predict_proba(df)[:, 1][0]
    print(f"athlete       : {args.athlete_id}  (week {args.week})")
    print(f"risk          : {proba:.4f}  ({proba * 100:.1f}%)")
    print(f"predicted     : {'INJURED' if proba >= threshold else 'healthy'}  "
          f"@ threshold {threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
