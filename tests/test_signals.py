import numpy as np
import pandas as pd
import pytest

from src.features import add_features
from src.labels import _trigger_mask
from src.signals import scan_all, scan_symbol


def _cfg(mode="relaxed", rsi_max=55.0, warmup=100):
    return {
        "labeling": {"trigger_mode": mode, "rsi_max": rsi_max},
        "execution": {"warmup_days": warmup},
    }


def _ohlcv(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes, "volume": 1.0,
    })


def _random_walk(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    return _ohlcv(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n)))


def _firing_df(mode="relaxed", rsi_max=55.0, min_idx=100):
    """Random-walk OHLCV truncated so the trigger fires on the last bar.

    Features are causal (no lookahead), so truncating after a firing bar
    leaves that bar's feature values — and the trigger — unchanged.
    """
    df = _random_walk()
    mask = _trigger_mask(add_features(df), mode, rsi_max)
    firing = np.where(mask.values)[0]
    firing = firing[firing >= min_idx]
    assert len(firing) > 0, "fixture random walk never fires the trigger"
    return df.iloc[: firing[0] + 1].reset_index(drop=True)


class TestScanSymbol:
    def test_returns_none_below_warmup(self):
        df = _ohlcv([100.0] * 50)
        assert scan_symbol(df, _cfg(warmup=100)) is None

    def test_returns_none_when_trigger_does_not_fire(self):
        # Steady uptrend: RSI is high on the last bar, the dip-buy can't fire
        df = _ohlcv(np.linspace(50, 150, 300))
        assert scan_symbol(df, _cfg()) is None

    @pytest.mark.parametrize("mode", ["relaxed", "strict"])
    def test_fires_when_last_bar_meets_trigger(self, mode):
        sig = scan_symbol(_firing_df(mode=mode), _cfg(mode=mode))
        assert sig is not None

    def test_signal_dict_has_expected_fields(self):
        df = _firing_df()
        sig = scan_symbol(df, _cfg())
        assert sig is not None
        assert {"date", "close", "rsi_14", "macd", "macd_signal"} <= set(sig)
        assert sig["close"] == pytest.approx(df["close"].iloc[-1])

    def test_only_last_bar_matters(self):
        # Append calm flat days after the firing bar → no longer firing
        df = _firing_df()
        flat = _ohlcv([float(df["close"].iloc[-1])] * 30)
        extended = pd.concat([df, flat], ignore_index=True)
        assert scan_symbol(extended, _cfg()) is None

    def test_respects_rsi_max_from_config(self):
        df = _firing_df(rsi_max=55.0)
        assert scan_symbol(df, _cfg(rsi_max=55.0)) is not None
        # An impossibly low threshold silences the same data
        assert scan_symbol(df, _cfg(rsi_max=1.0)) is None


class TestScanAll:
    def test_empty_when_no_symbol_fires(self):
        data = {"AAA": _ohlcv(np.linspace(50, 150, 300))}
        result = scan_all(data, _cfg())
        assert result.empty

    def test_collects_firing_symbols_indexed_by_ticker(self):
        data = {
            "FIRE": _firing_df(),
            "CALM": _ohlcv(np.linspace(50, 150, 300)),
        }
        result = scan_all(data, _cfg())
        assert list(result.index) == ["FIRE"]
        assert "close" in result.columns
