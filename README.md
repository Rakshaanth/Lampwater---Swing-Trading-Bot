# Lampwater — Swing Trading Bot

ML-driven paper trading pipeline using Alpaca. Pulls historical OHLCV data, engineers technical features, labels trades with an asymmetric profit/loss rule, trains an XGBoost classifier, and executes paper trades when model confidence exceeds a threshold.

## Progress

- [x] Project structure and config (`config/config.yaml`, `config/tickers.txt`)
- [x] Data pipeline (`src/data.py`, `scripts/fetch_data.py`)
- [x] Feature engineering (`src/features.py`) — RSI-14, MACD (+cross), SMA-50/200
- [x] Signal trigger + labeling (`src/labels.py`) — `scripts/build_dataset.py` still TODO
- [x] **Rule-based v1 live** (`src/signals.py`, `src/execute.py`, `scripts/run_daily.py`) —
      scans the latest bar for the trigger, sizes positions, submits Alpaca paper
      orders, logs to `logs/trades_{date}.jsonl`. Dry run: `python scripts/run_daily.py --no-trade`
      (add `--cached` to use `data/raw/` instead of fetching).
- [ ] XGBoost training + eval (`src/train.py`, `scripts/train_model.py`)
- [ ] ML inference path (`src/predict.py`) — v1 trades the rule trigger directly, no model
- [ ] Smoke test (`scripts/smoke_test.py`)

### Heads-up: the original trigger never fires

Empirically (5y × 34 tickers, ~40k bar-days), RSI-14 **never** drops below ~37
while price is above the 50-day SMA — so `RSI < 35 AND above_sma50` yields zero
candidates in strict *and* relaxed mode. The RSI cutoff is now configurable as
`labeling.rsi_max` in `config.yaml`, defaulting to 55 with `trigger_mode: relaxed`
(~1.7k historical candidates, 9.4% hit +8% before −5% within 5 days, vs 8.4%
baseline on all bars).
