"""Rule-based signal scanner: applies the entry trigger to the latest bar.

Reuses the same trigger as offline labeling (labels._trigger_mask) so live
signals and training labels can never drift apart.
"""
import pandas as pd

from src.features import add_features
from src.labels import _trigger_mask


def scan_symbol(df: pd.DataFrame, cfg: dict) -> dict | None:
    """Evaluate the entry trigger on the most recent bar of one symbol's OHLCV.

    Returns a signal dict if the trigger fires, None otherwise (including when
    there are fewer than warmup_days bars, where features are unreliable).
    """
    if len(df) < cfg["execution"]["warmup_days"]:
        return None
    feat = add_features(df)
    mask = _trigger_mask(feat, cfg["labeling"]["trigger_mode"], cfg["labeling"].get("rsi_max", 35.0))
    if not bool(mask.iloc[-1]):
        return None
    last = feat.iloc[-1]
    return {
        "date": str(feat.index[-1])[:10],
        "close": float(last["close"]),
        "rsi_14": float(last["rsi_14"]),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
    }


def scan_all(data: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Scan many symbols; returns a DataFrame indexed by symbol, empty if none fire."""
    rows = {}
    for symbol, df in data.items():
        sig = scan_symbol(df, cfg)
        if sig is not None:
            rows[symbol] = sig
    return pd.DataFrame.from_dict(rows, orient="index")
