import numpy as np
import pandas as pd
import pytest

from src.features import add_features
from src.labels import _forward_label, _trigger_mask, label_candidates


def _make_df(closes, highs=None, lows=None, opens=None):
    """Build a minimal OHLCV DataFrame for testing."""
    n = len(closes)
    closes = np.array(closes, dtype=float)
    if opens is None:
        opens = closes
    if highs is None:
        highs = closes
    if lows is None:
        lows = closes
    return pd.DataFrame({
        "open": np.array(opens, dtype=float),
        "high": np.array(highs, dtype=float),
        "low": np.array(lows, dtype=float),
        "close": np.array(closes, dtype=float),
        "volume": np.ones(n),
    })


def _cfg(horizon=5, tp=0.08, sl=0.05, mode="strict"):
    return {"labeling": {"horizon": horizon, "take_profit": tp, "stop_loss": sl, "trigger_mode": mode}}


# ---------------------------------------------------------------------------
# _forward_label
# ---------------------------------------------------------------------------

class TestForwardLabel:
    def test_tp_hit_before_sl(self):
        # Entry at open[1]=100; tp=108, sl=95
        # Day 1 (i+1): high=109 → TP hit
        closes = [100.0] * 10
        opens  = [100.0] * 10
        highs  = [100.0] * 10
        lows   = [100.0] * 10
        highs[1] = 109.0  # TP hit on first forward bar
        df = _make_df(closes, highs=highs, lows=lows, opens=opens)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 1

    def test_sl_hit_before_tp(self):
        closes = [100.0] * 10
        opens  = [100.0] * 10
        highs  = [100.0] * 10
        lows   = [100.0] * 10
        lows[1] = 94.0  # SL hit on first forward bar
        df = _make_df(closes, highs=highs, lows=lows, opens=opens)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 0

    def test_same_day_ambiguity_is_zero(self):
        # Both TP and SL hit on the same bar → label 0 (conservative)
        closes = [100.0] * 10
        opens  = [100.0] * 10
        highs  = [100.0] * 10
        lows   = [100.0] * 10
        highs[1] = 110.0  # above TP
        lows[1]  = 90.0   # below SL
        df = _make_df(closes, highs=highs, lows=lows, opens=opens)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 0

    def test_neither_hits_within_horizon(self):
        closes = [100.0] * 10
        df = _make_df(closes)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 0

    def test_tp_hit_on_last_bar_of_horizon(self):
        closes = [100.0] * 10
        opens  = [100.0] * 10
        highs  = [100.0] * 10
        lows   = [100.0] * 10
        highs[5] = 109.0  # hit exactly at horizon boundary
        df = _make_df(closes, highs=highs, lows=lows, opens=opens)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 1

    def test_sl_hit_on_earlier_bar_wins_over_tp_later(self):
        # SL on day 2, TP on day 3 — SL should win
        closes = [100.0] * 10
        opens  = [100.0] * 10
        highs  = [100.0] * 10
        lows   = [100.0] * 10
        lows[2]  = 94.0   # SL on day 2
        highs[3] = 109.0  # TP on day 3 (shouldn't matter)
        df = _make_df(closes, highs=highs, lows=lows, opens=opens)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 0

    def test_last_horizon_rows_are_nan(self):
        closes = [100.0] * 10
        df = _make_df(closes)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[-5:].isna().all()

    def test_resolved_rows_are_not_nan(self):
        closes = [100.0] * 10
        df = _make_df(closes)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[:-5].notna().all()

    def test_entry_uses_next_day_open_not_close(self):
        # close[0]=100, open[1]=50; TP threshold should be 50*1.08=54, not 100*1.08=108
        closes = [100.0] * 10
        opens  = [100.0] * 10
        highs  = [100.0] * 10
        lows   = [100.0] * 10
        opens[1] = 50.0
        highs[1] = 55.0  # above 54 (TP based on open[1]=50), below 108 (TP based on close[0]=100)
        df = _make_df(closes, highs=highs, lows=lows, opens=opens)
        result = _forward_label(df, horizon=5, tp_pct=0.08, sl_pct=0.05)
        assert result.iloc[0] == 1  # would be 0 if using close[0] as entry


# ---------------------------------------------------------------------------
# _trigger_mask
# ---------------------------------------------------------------------------

class TestTriggerMask:
    def _base_df_with_features(self, n=250):
        close = pd.Series(np.linspace(50, 150, n), dtype=float)
        df = pd.DataFrame({
            "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1.0,
        })
        return add_features(df)

    def test_strict_requires_all_three_conditions(self):
        df = self._base_df_with_features()
        # Force one row to meet all strict conditions
        idx = 210
        df.loc[idx, "rsi_14"] = 30.0
        df.loc[idx, "macd_cross"] = 1
        df.loc[idx, "above_sma50"] = True
        mask = _trigger_mask(df, "strict")
        assert mask.iloc[idx]

    def test_strict_fails_if_rsi_above_35(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 40.0  # RSI too high
        df.loc[idx, "macd_cross"] = 1
        df.loc[idx, "above_sma50"] = True
        mask = _trigger_mask(df, "strict")
        assert not mask.iloc[idx]

    def test_strict_fails_if_no_macd_cross(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 30.0
        df.loc[idx, "macd_cross"] = 0  # no cross
        df.loc[idx, "above_sma50"] = True
        mask = _trigger_mask(df, "strict")
        assert not mask.iloc[idx]

    def test_strict_fails_if_below_sma50(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 30.0
        df.loc[idx, "macd_cross"] = 1
        df.loc[idx, "above_sma50"] = False
        mask = _trigger_mask(df, "strict")
        assert not mask.iloc[idx]

    def test_relaxed_uses_macd_line_not_cross(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 30.0
        df.loc[idx, "macd_cross"] = 0    # no crossover, but:
        df.loc[idx, "macd"] = 1.0        # macd > signal
        df.loc[idx, "macd_signal"] = 0.5
        df.loc[idx, "above_sma50"] = True
        mask = _trigger_mask(df, "relaxed")
        assert mask.iloc[idx]

    def test_relaxed_still_requires_rsi_and_sma50(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 40.0    # RSI too high
        df.loc[idx, "macd"] = 1.0
        df.loc[idx, "macd_signal"] = 0.5
        df.loc[idx, "above_sma50"] = True
        mask = _trigger_mask(df, "relaxed")
        assert not mask.iloc[idx]

    def test_unknown_mode_raises(self):
        df = self._base_df_with_features()
        with pytest.raises(ValueError, match="Unknown trigger_mode"):
            _trigger_mask(df, "bogus")

    def test_rsi_max_is_configurable(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 50.0  # above default 35, below 55
        df.loc[idx, "macd_cross"] = 1
        df.loc[idx, "above_sma50"] = True
        assert not _trigger_mask(df, "strict").iloc[idx]
        assert _trigger_mask(df, "strict", rsi_max=55.0).iloc[idx]

    def test_label_candidates_reads_rsi_max_from_cfg(self):
        df = self._base_df_with_features()
        idx = 210
        df.loc[idx, "rsi_14"] = 50.0
        df.loc[idx, "macd_cross"] = 1
        df.loc[idx, "above_sma50"] = True
        cfg = _cfg()
        cfg["labeling"]["rsi_max"] = 55.0
        result = label_candidates(df, cfg)
        assert idx in result.index


# ---------------------------------------------------------------------------
# label_candidates (integration)
# ---------------------------------------------------------------------------

class TestLabelCandidates:
    def _make_featured_df(self, n=260):
        close = pd.Series(np.linspace(50, 150, n), dtype=float)
        df = pd.DataFrame({
            "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1.0,
        })
        return add_features(df)

    def test_returns_dataframe_with_target_column(self):
        df = self._make_featured_df()
        result = label_candidates(df, _cfg())
        assert "target" in result.columns

    def test_target_is_binary_int(self):
        df = self._make_featured_df()
        result = label_candidates(df, _cfg())
        assert result["target"].isin([0, 1]).all()

    def test_no_nans_in_target(self):
        df = self._make_featured_df()
        result = label_candidates(df, _cfg())
        assert result["target"].notna().all()

    def test_candidates_excluded_from_last_horizon_rows(self):
        df = self._make_featured_df()
        cfg = _cfg(horizon=5)
        result = label_candidates(df, cfg)
        if len(result) > 0:
            # No candidate should come from the last 5 rows
            assert result.index.max() <= df.index[-6]

    def test_unknown_trigger_mode_raises(self):
        df = self._make_featured_df()
        with pytest.raises(ValueError):
            label_candidates(df, _cfg(mode="invalid"))

    def test_relaxed_mode_produces_more_candidates_than_strict(self):
        # Relaxed trigger (macd > signal) is broader than strict (macd_cross)
        df = self._make_featured_df(n=500)
        # Force some rows to have low RSI and above_sma50 so triggers can fire
        df.loc[100:200, "rsi_14"] = 30.0
        df.loc[100:200, "above_sma50"] = True
        df.loc[100:200, "macd"] = 1.0
        df.loc[100:200, "macd_signal"] = 0.5
        df.loc[150, "macd_cross"] = 1  # only one strict cross

        strict = label_candidates(df, _cfg(mode="strict"))
        relaxed = label_candidates(df, _cfg(mode="relaxed"))
        assert len(relaxed) >= len(strict)
