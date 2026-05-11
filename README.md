# Lampwater — Swing Trading Bot

ML-driven paper trading pipeline using Alpaca. Pulls historical OHLCV data, engineers technical features, labels trades with an asymmetric profit/loss rule, trains an XGBoost classifier, and executes paper trades when model confidence exceeds a threshold.

## Progress

- [x] Project structure and config (`config/config.yaml`, `config/tickers.txt`)
- [x] Data pipeline (`src/data.py`, `scripts/fetch_data.py`)
- [ ] Feature engineering (`src/features.py`)
- [ ] Signal trigger + labeling (`src/labels.py`, `scripts/build_dataset.py`)
- [ ] XGBoost training + eval (`src/train.py`, `scripts/train_model.py`)
- [ ] Paper trading execution (`src/predict.py`, `src/execute.py`, `scripts/run_daily.py`)
- [ ] Smoke test (`scripts/smoke_test.py`)
