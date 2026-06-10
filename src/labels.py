import numpy as np
import pandas as pd


def _forward_label(df: pd.DataFrame, horizon: int, tp_pct: float, sl_pct: float) -> pd.Series:
    """
    For each row i, compute the trade outcome using open[i+1] as the entry price.
    Checks high/low on days i+1 through i+horizon.

    Returns NaN for the last `horizon` rows (incomplete forward window).
    Same-day TP+SL hit → 0 (conservative; daily bars can't resolve intraday order).
    """
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(df)
    target = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = open_[i + 1]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        result = 0
        for k in range(1, horizon + 1):
            j = i + k
            tp_hit = high[j] >= tp
            sl_hit = low[j] <= sl
            if tp_hit and sl_hit:
                result = 0
                break
            elif tp_hit:
                result = 1
                break
            elif sl_hit:
                result = 0
                break
        target[i] = result

    return pd.Series(target, index=df.index)


def _trigger_mask(df: pd.DataFrame, mode: str, rsi_max: float = 35.0) -> pd.Series:
    """Returns boolean mask of rows that meet the entry trigger.

    Note: empirically (5y × 34 mid-caps), RSI-14 never drops below ~37 while
    price is above the 50d SMA, so rsi_max below ~50 yields zero candidates
    in either mode. Configure via labeling.rsi_max in config.yaml.
    """
    if mode == "strict":
        return (df["rsi_14"] < rsi_max) & (df["macd_cross"] == 1) & df["above_sma50"]
    elif mode == "relaxed":
        return (df["rsi_14"] < rsi_max) & (df["macd"] > df["macd_signal"]) & df["above_sma50"]
    else:
        raise ValueError(f"Unknown trigger_mode: {mode!r}. Expected 'strict' or 'relaxed'.")


def label_candidates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Given a DataFrame with OHLCV + features (from add_features), returns a
    filtered DataFrame of candidate rows with a `target` column (1 = TP hit
    before SL within horizon bars, 0 = SL hit first / neither hit).

    Rows with an incomplete forward window (last `horizon` rows) are excluded.
    """
    lab = cfg["labeling"]
    horizon: int = lab["horizon"]
    tp_pct: float = lab["take_profit"]
    sl_pct: float = lab["stop_loss"]
    mode: str = lab["trigger_mode"]
    rsi_max: float = lab.get("rsi_max", 35.0)

    all_targets = _forward_label(df, horizon, tp_pct, sl_pct)

    mask = _trigger_mask(df, mode, rsi_max)
    # Also exclude rows whose forward window is incomplete (target is NaN)
    resolved = all_targets.notna()

    candidates = df[mask & resolved].copy()
    candidates["target"] = all_targets[mask & resolved].astype(int)

    return candidates
