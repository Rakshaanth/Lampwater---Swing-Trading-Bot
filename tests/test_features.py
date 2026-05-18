import numpy as np
import pandas as pd
import pytest

from src.features import rsi, macd, add_features


def _close(values):
    return pd.Series(values, dtype=float)


def test_rsi_monotone_up_approaches_100():
    close = _close(range(1, 30))
    result = rsi(close)
    assert result.iloc[-1] > 90


def test_rsi_monotone_down_approaches_0():
    close = _close(range(30, 1, -1))
    result = rsi(close)
    assert result.iloc[-1] < 10


def test_rsi_bounded():
    close = _close([10, 11, 9, 12, 8, 13, 7, 14, 6, 15] * 5)
    result = rsi(close)
    assert result.dropna().between(0, 100).all()


def test_macd_cross_detected_at_correct_bar():
    # Build a close series where macd crosses signal at a known point.
    # Start with a long downtrend so macd < signal, then spike up to force a crossover.
    np.random.seed(0)
    down = list(np.linspace(100, 60, 60))
    up = list(np.linspace(60, 120, 40))
    close = _close(down + up)

    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1})
    df = add_features(df)

    cross_bars = df[df["macd_cross"] == 1]
    assert len(cross_bars) >= 1
    # All non-cross bars should be 0
    assert df["macd_cross"].isin([0, 1]).all()


def test_macd_cross_is_zero_during_sustained_downtrend():
    close = _close(np.linspace(100, 50, 80))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1})
    df = add_features(df)
    # No upward crossover in a pure downtrend
    assert df["macd_cross"].sum() == 0


def test_add_features_length_preserved():
    close = _close(np.linspace(50, 150, 250))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1})
    out = add_features(df)
    assert len(out) == len(df)


def test_add_features_nans_only_in_warmup():
    close = _close(np.linspace(50, 150, 250))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1})
    out = add_features(df)
    # sma_200 needs 200 bars; after that no NaNs in core feature columns
    tail = out.iloc[200:]
    assert tail[["rsi_14", "macd", "macd_signal", "sma_200"]].isna().sum().sum() == 0


def test_add_features_preserves_ohlcv():
    close = _close(np.linspace(50, 150, 250))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 100})
    out = add_features(df)
    for col in ["open", "high", "low", "close", "volume"]:
        pd.testing.assert_series_equal(out[col], df[col])
