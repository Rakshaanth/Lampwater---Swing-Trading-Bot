import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def _client() -> StockHistoricalDataClient:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    return StockHistoricalDataClient(key, secret)


def fetch_ohlcv(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = _client()
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
    bars = client.get_stock_bars(req)
    df = bars.df
    if df.empty:
        return df
    # bars.df has a MultiIndex (symbol, timestamp) — drop the symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    df.index.name = "date"
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def fetch_all(symbols: list[str], start: datetime, end: datetime, out_dir: str, sleep_sec: float = 0.3):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, symbol in enumerate(symbols):
        dest = out / f"{symbol}.parquet"
        if dest.exists():
            print(f"[{i+1}/{len(symbols)}] {symbol} already cached, skipping")
            continue
        try:
            df = fetch_ohlcv(symbol, start, end)
            if df.empty:
                print(f"[{i+1}/{len(symbols)}] {symbol} — no data returned")
            else:
                df.to_parquet(dest)
                print(f"[{i+1}/{len(symbols)}] {symbol} — {len(df)} rows saved")
        except Exception as e:
            print(f"[{i+1}/{len(symbols)}] {symbol} — error: {e}")
        time.sleep(sleep_sec)


def load_ohlcv(symbol: str, raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}. Run fetch_data.py first.")
    return pd.read_parquet(path)
