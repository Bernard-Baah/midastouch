"""
MidasTouch - Mean Reversion Strategy
RSI + Bollinger Band mean-reversion signals.

Signal Logic:
    BUY  when: RSI < 35, price below lower BB, volume above average
    SELL when: RSI > 65, price above upper BB
    Score range: -1.0 (strong sell) to +1.0 (strong buy)
"""

import logging
from typing import Tuple

import pandas as pd

from core.indicators import get_latest_row, validate_indicators

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Generate a mean-reversion signal from the latest indicator row.

    Score components:
        +0.40  RSI < 35 (oversold)
        +0.35  price below lower Bollinger Band
        +0.25  volume above 20-period average
        Mirror conditions for sell signals.

    Args:
        df: DataFrame with indicator columns already calculated.

    Returns:
        Tuple of (score: float, reason: str)
    """
    row = get_latest_row(df)
    if row is None or not validate_indicators(row):
        logger.debug("MeanReversion: no valid data, returning hold")
        return 0.0, 'hold'

    return _score_row(row)


def _score_row(row: pd.Series) -> Tuple[float, str]:
    """
    Score a single indicator row for mean-reversion opportunities.

    Args:
        row: pd.Series with indicator values

    Returns:
        (score, direction_string)
    """
    close       = float(row['close'])
    rsi         = float(row['rsi_14'])
    bb_upper    = float(row['bb_upper'])
    bb_lower    = float(row['bb_lower'])
    volume      = float(row['volume'])
    vol_sma     = float(row['volume_sma_20'])

    score = 0.0
    reasons = []

    high_volume = vol_sma > 0 and volume > vol_sma

    # ── Oversold (buy) ────────────────────────────────────────────────────────
    if rsi < 35:
        score += 0.40
        reasons.append('RSI_oversold')
    elif rsi < 40:
        score += 0.20
        reasons.append('RSI_near_oversold')

    if close < bb_lower:
        score += 0.35
        reasons.append('below_BB_lower')
    elif close < (bb_lower * 1.005):   # within 0.5% of lower band
        score += 0.15
        reasons.append('near_BB_lower')

    if high_volume and score > 0:
        score += 0.25
        reasons.append('high_volume_confirmation')

    # ── Overbought (sell) ─────────────────────────────────────────────────────
    if rsi > 65:
        score -= 0.40
        reasons.append('RSI_overbought')
    elif rsi > 60:
        score -= 0.20
        reasons.append('RSI_near_overbought')

    if close > bb_upper:
        score -= 0.35
        reasons.append('above_BB_upper')
    elif close > (bb_upper * 0.995):
        score -= 0.15
        reasons.append('near_BB_upper')

    score = max(-1.0, min(1.0, score))
    direction = 'buy' if score > 0 else ('sell' if score < 0 else 'hold')

    logger.debug(
        "MeanReversion score=%.3f direction=%s reasons=%s",
        score, direction, reasons
    )
    return score, direction


def score_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculate mean-reversion scores for every row — useful for backtesting.

    Args:
        df: Full indicator DataFrame

    Returns:
        pd.Series of float scores
    """
    scores = []
    for _, row in df.iterrows():
        try:
            s, _ = _score_row(row)
        except Exception:  # pylint: disable=broad-except
            s = 0.0
        scores.append(s)
    return pd.Series(scores, index=df.index, name='mr_score')
