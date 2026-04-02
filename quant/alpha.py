"""
MidasTouch Quant Layer — Alpha Signal Extraction
Reads engineered features and produces normalised per-factor alpha signals.
Each signal is in [-1, +1]:
  +1 = strong bullish
  -1 = strong bearish
   0 = neutral
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configurable factor weights — tune here without touching ensemble.py
FACTOR_WEIGHTS = {
    "momentum":    0.30,
    "mean_rev":    0.25,
    "volatility":  0.20,
    "volume":      0.25,
}


def extract_alpha_signals(df: pd.DataFrame) -> dict[str, float]:
    """
    Extract the four alpha signals from the latest row of a feature DataFrame.

    Args:
        df: DataFrame produced by quant.features.add_all_features()

    Returns:
        dict with keys: momentum, mean_rev, volatility, volume
        All values in [-1, +1], or 0.0 on error/NaN.
    """
    if df.empty:
        return _zero_signals()

    row = df.iloc[-1]

    def safe(key: str) -> float:
        val = row.get(key, 0.0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(np.clip(val, -1.0, 1.0))

    signals = {
        "momentum":   safe("momentum"),
        "mean_rev":   safe("mean_reversion"),
        "volatility": safe("volatility_signal"),
        "volume":     safe("volume_spike"),
    }

    logger.debug("Alpha signals: %s", signals)
    return signals


def _zero_signals() -> dict[str, float]:
    return {"momentum": 0.0, "mean_rev": 0.0, "volatility": 0.0, "volume": 0.0}
