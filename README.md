# End-to-End Injury Risk ML Pipeline

A production-style machine learning system that predicts athlete injury
risk from weekly training, sleep, and recovery data. Built with
scikit-learn, designed around a single sklearn `Pipeline` so feature
engineering, preprocessing, and the classifier are trained, evaluated,
and serialized as one unit. Served via FastAPI.



## Project layout

```
injury_risk/
├── src/
│   ├── data.py             # synthetic generator + CSV loader + athlete split
│   ├── features.py         # FeatureEngineer (sklearn-compatible)
│   ├── models.py           # build_pipeline + model factory
│   ├── evaluation.py       # ROC, PR, F1, Brier, GroupKFold CV harness
│   ├── viz.py              # ROC/PR/calibration/confusion/importance plots
│   └── api.py              # FastAPI service + browser console
├── scripts/
│   ├── train.py            # full pipeline: load → split → CV → fit → save
│   ├── predict.py          # CLI prediction (single record or CSV)
│   └── smoke_test.py       # pytest-free pipeline validator
├── tests/
│   ├── test_data_and_features.py
│   ├── test_pipeline_and_evaluation.py
│   └── test_api.py
├── data/                   # gitignored - real data goes here
├── models/                 # gitignored - trained pipelines
├── figures/                # gitignored - diagnostic plots
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Validate the pipeline end-to-end
python scripts/smoke_test.py

# 3. Train on synthetic data (no dataset required)
python scripts/train.py --model gradient_boosting

# 4. Predict for a single athlete-week
python scripts/predict.py --athlete-id A0001 --week 12 --age 24 \
    --weekly-load 2200 --sleep 6.5 --soreness 6 --rpe 7.5

# 5. Or serve via FastAPI
uvicorn src.api:app --reload
# then visit http://127.0.0.1:8000/
```

## Training on real data

The expected CSV schema (one row per athlete-week):

| column                 | required? | dtype  | notes                          |
| ---------------------- | --------- | ------ | ------------------------------ |
| `athlete_id`           | ✓         | string | grouping key                   |
| `week`                 | ✓         | int    | sequential per athlete         |
| `age`                  | ✓         | float  |                                |
| `weekly_load`          | ✓         | float  | sum of training session loads  |
| `sleep_hours`          | ✓         | float  | weekly average                 |
| `soreness`             | ✓         | int    | 1–10 self-report               |
| `injured`              | ✓         | int    | label, 0/1, for THIS week      |
| `sex`                  |           | str    | "M"/"F", optional              |
| `position`             |           | str    | categorical, optional          |
| `rpe_avg`              |           | float  | rate of perceived exertion 1–10|
| `sessions_count`       |           | int    | sessions trained that week     |
| `sprint_distance_km`   |           | float  |                                |
| `prior_injuries`       |           | int    | career-to-date injury count    |

Then:

```bash
python scripts/train.py --data data/athletes.csv --model gradient_boosting
```


## Evaluation

The training script reports several metrics because no single number
captures imbalanced classification well:

- **ROC-AUC**: threshold-independent ranking quality
- **PR-AUC**: more informative than ROC for rare events
- **F1**: balanced precision/recall at the chosen threshold
- **Brier score**: mean squared error of predicted probabilities, a proxy for calibration
- **Confusion matrix** at a tuned decision threshold

The threshold itself is tuned on a held-out CV fold (not on the test
set) by maximizing F1. Default 0.5 is rarely optimal under class
imbalance.

## API reference

| Method | Path              | Purpose                                  |
| ------ | ----------------- | ---------------------------------------- |
| GET    | `/health`         | Liveness                                 |
| GET    | `/info`           | Pipeline metadata + training metrics     |
| POST   | `/predict`        | Predict for one athlete-week             |
| POST   | `/predict/batch`  | Predict for many records at once         |
| GET    | `/docs`           | Auto-generated OpenAPI docs              |
| GET    | `/`               | Tiny browser console                     |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{
    "target": {
      "athlete_id": "A0001",
      "week": 12,
      "age": 24,
      "weekly_load": 2200,
      "sleep_hours": 6.5,
      "soreness": 6,
      "rpe_avg": 7.5,
      "prior_injuries": 1
    },
    "history": [],
    "threshold": 0.5
  }'
```

```json
{
  "athlete_id": "A0001",
  "week": 12,
  "risk_probability": 0.247,
  "predicted_injured": 0,
  "threshold": 0.5
}
```

## Tests

```bash
pytest -q
```

Three test files:

- `test_data_and_features.py` — synthetic generator, CSV loader, athlete
  splits, feature engineer (including the no-label-leakage regression
  test)
- `test_pipeline_and_evaluation.py` — end-to-end pipeline for all three
  models, save/load roundtrip, evaluation metrics, threshold tuning,
  grouped CV
- `test_api.py` — FastAPI endpoints via TestClient (skipped if FastAPI
  is missing)



