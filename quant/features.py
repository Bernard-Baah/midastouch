"""
MidasTouch Quant Layer — Feature Engineering
Computes raw features from OHLCV DataFrames.
All functions are pure: DataFrame in, DataFrame out.
No side effects.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame, windows: list[int] = [1, 5]) -> pd.DataFrame:
    """
    Add pct_return_Nd columns for each window in `windows`.
    e.g. windows=[1,5] → pct_return_1d, pct_return_5d
    """
    out = df.copy()
    for w in windows:
        col = f"pct_return_{w}d"
        out[col] = out["close"].pct_change(w)
    return out


def compute_momentum(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Time-series momentum: rolling return over `window` periods.
    Normalised to [-1, 1] via tanh.
    """
    out = df.copy()
    raw = out["close"].pct_change(window)
    out["momentum"] = np.tanh(raw * 10)   # scale before squash
    return out


def compute_mean_reversion(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Z-score of close vs rolling mean: (close - mean) / std.
    Negative z → price below mean → mean-reversion BUY signal (negated).
    Clipped to [-3, 3] then normalised to [-1, 1].
    """
    out = df.copy()
    mu  = out["close"].rolling(window).mean()
    sig = out["close"].rolling(window).std()
    z   = (out["close"] - mu) / sig.replace(0, np.nan)
    z   = z.clip(-3, 3)
    out["mean_reversion"] = -(z / 3)   # invert: below mean → positive signal
    return out


def compute_rolling_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Normalised rolling volatility of returns.
    High vol → signal = 1 (useful for breakout strategies).
    Low vol  → signal approaches 0.
    """
    out = df.copy()
    ret_vol = out["close"].pct_change().rolling(window).std()
    # Normalise: rank against own 60-period history
    hist_max = ret_vol.rolling(60).max().replace(0, np.nan)
    out["volatility_signal"] = (ret_vol / hist_max).clip(0, 1)
    return out


def compute_volume_spike(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Relative volume vs rolling average.
    Spike ratio clipped to [0, 3], normalised to [0, 1].
    """
    out = df.copy()
    avg_vol = out["volume"].rolling(window).mean().replace(0, np.nan)
    ratio   = (out["volume"] / avg_vol).clip(0, 3)
    out["volume_spike"] = ratio / 3
    return out


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: apply all feature functions in sequence.
    Returns DataFrame with all quant feature columns added.
    """
    df = compute_returns(df)
    df = compute_momentum(df)
    df = compute_mean_reversion(df)
    df = compute_rolling_volatility(df)
    df = compute_volume_spike(df)
    return df
