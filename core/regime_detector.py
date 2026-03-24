"""
MidasTouch - Market Regime Detector
Classifies current market conditions to guide strategy weighting.

Regimes:
    'bull'             - Price above 200 EMA, RSI trending upward, low-moderate vol
    'bear'             - Price below 200 EMA, RSI in low territory
    'sideways'         - Price near 200 EMA, RSI in 40-60 range
    'high_volatility'  - ATR significantly above its 30-period average
"""

import logging
from typing import Optional

import pandas as pd

from core.indicators import get_latest_row, validate_indicators

logger = logging.getLogger(__name__)

REGIME_LABELS = ('bull', 'bear', 'sideways', 'high_volatility')


def detect_regime(df: pd.DataFrame) -> str:
    """
    Determine the current market regime from the most recent indicator row.

    Detection priority:
        1. High volatility (overrides directional regimes)
        2. Bull (price > EMA200, RSI > 50)
        3. Bear (price < EMA200, RSI < 50)
        4. Sideways (default fallback)

    Args:
        df: DataFrame that has already had calculate_indicators() applied.

    Returns:
        One of: 'bull', 'bear', 'sideways', 'high_volatility'
    """
    row = get_latest_row(df)
    if row is None or not validate_indicators(row):
        logger.warning("Regime detector: invalid/missing indicators — defaulting to 'sideways'")
        return 'sideways'

    regime = _classify(row)
    logger.debug(
        "Regime detected: %s | price=%.4f ema200=%.4f rsi=%.1f atr=%.4f atr_avg=%.4f",
        regime,
        row['close'],
        row['ema_200'],
        row['rsi_14'],
        row['atr_14'],
        row['atr_avg_30'],
    )
    return regime


def _classify(row: pd.Series) -> str:
    """
    Core classification logic extracted for testability.

    Args:
        row: Single indicator row

    Returns:
        Regime string
    """
    close    = float(row['close'])
    ema_200  = float(row['ema_200'])
    rsi      = float(row['rsi_14'])
    atr      = float(row['atr_14'])
    atr_avg  = float(row['atr_avg_30'])

    # ── 1. High Volatility ────────────────────────────────────────────────────
    if atr_avg > 0 and atr > atr_avg * 1.5:
        return 'high_volatility'

    # ── 2. RSI extremes override ──────────────────────────────────────────────
    if rsi > 70:
        # Strong uptrend momentum
        return 'bull' if close >= ema_200 else 'high_volatility'

    if rsi < 30:
        # Strong downtrend momentum
        return 'bear'

    # ── 3. Price vs 200 EMA with RSI confirmation ─────────────────────────────
    if close > ema_200 and rsi >= 50:
        return 'bull'

    if close < ema_200 and rsi <= 50:
        return 'bear'

    # ── 4. Default: sideways / ranging ───────────────────────────────────────
    return 'sideways'


def regime_summary(df: pd.DataFrame) -> dict:
    """
    Return a detailed regime analysis dictionary for reporting/dashboard.

    Args:
        df: Indicator DataFrame

    Returns:
        Dict with regime label and supporting metrics
    """
    row = get_latest_row(df)
    if row is None:
        return {'regime': 'unknown', 'error': 'No data'}

    regime = detect_regime(df)
    return {
        'regime':       regime,
        'close':        float(row.get('close', 0)),
        'ema_200':      float(row.get('ema_200', 0)),
        'rsi_14':       float(row.get('rsi_14', 50)),
        'atr_14':       float(row.get('atr_14', 0)),
        'atr_avg_30':   float(row.get('atr_avg_30', 0)),
        'above_ema200': float(row.get('close', 0)) > float(row.get('ema_200', 0)),
    }
