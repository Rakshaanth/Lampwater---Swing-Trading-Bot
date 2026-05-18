import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI: EMA with alpha = 1/period."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, SMA columns in-place on a copy. Preserves OHLCV columns."""
    df = df.copy()
    df["rsi_14"] = rsi(df["close"], period=14)
    macd_line, signal_line, _ = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    prev_below = df["macd"].shift(1) <= df["macd_signal"].shift(1)
    curr_above = df["macd"] > df["macd_signal"]
    df["macd_cross"] = (prev_below & curr_above).astype(int)
    df["sma_200"] = sma(df["close"], period=200)
    df["above_sma200"] = df["close"] > df["sma_200"]
    return df
