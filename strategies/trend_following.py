"""
MidasTouch - Trend Following Strategy
EMA crossover + momentum-based signals.

Signal Logic:
    BUY  when: EMA9 > EMA21 > EMA50, RSI in 50-65, MACD line > signal, price above EMA200
    SELL when: EMA9 < EMA21, RSI > 70 (overbought) or bearish MACD crossover
    Score range: -1.0 (strong sell) to +1.0 (strong buy)
"""

import logging
from typing import Optional, Tuple

import pandas as pd

from core.indicators import get_latest_row, validate_indicators

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Generate a trend-following signal from the latest indicator row.

    The score is built additively from sub-conditions:
        +0.30  EMA9 > EMA21 > EMA50 (aligned uptrend)
        +0.20  price above EMA200
        +0.25  MACD line above signal line
        +0.25  RSI in 50-65 zone (momentum without overbought)

    Mirror conditions produce equivalent negative scores.

    Args:
        df: DataFrame with indicator columns already calculated.

    Returns:
        Tuple of (score: float, reason: str)
        score is clamped to [-1.0, +1.0]
    """
    row = get_latest_row(df)
    if row is None or not validate_indicators(row):
        logger.debug("TrendFollowing: no valid data, returning hold")
        return 0.0, 'hold'

    return _score_row(row)


def _score_row(row: pd.Series) -> Tuple[float, str]:
    """
    Score a single indicator row.

    Args:
        row: pd.Series with all indicator values

    Returns:
        (score, direction_string)
    """
    close      = float(row['close'])
    ema_9      = float(row['ema_9'])
    ema_21     = float(row['ema_21'])
    ema_50     = float(row['ema_50'])
    ema_200    = float(row['ema_200'])
    rsi        = float(row['rsi_14'])
    macd_line  = float(row['macd_line'])
    macd_sig   = float(row['macd_signal'])

    score = 0.0
    reasons = []

    # ── EMA alignment ────────────────────────────────────────────────────────
    if ema_9 > ema_21 > ema_50:
        score += 0.30
        reasons.append('EMA_bull_aligned')
    elif ema_9 < ema_21 < ema_50:
        score -= 0.30
        reasons.append('EMA_bear_aligned')

    # ── Price vs EMA200 ───────────────────────────────────────────────────────
    if close > ema_200:
        score += 0.20
        reasons.append('above_EMA200')
    else:
        score -= 0.20
        reasons.append('below_EMA200')

    # ── MACD ─────────────────────────────────────────────────────────────────
    if macd_line > macd_sig:
        score += 0.25
        reasons.append('MACD_bull')
    else:
        score -= 0.25
        reasons.append('MACD_bear')

    # ── RSI momentum / overbought ─────────────────────────────────────────────
    if 50.0 <= rsi <= 65.0:
        score += 0.25
        reasons.append('RSI_momentum')
    elif rsi > 70.0:
        score -= 0.25
        reasons.append('RSI_overbought')
    elif rsi < 40.0:
        score -= 0.15
        reasons.append('RSI_weak')

    score = max(-1.0, min(1.0, score))
    direction = 'buy' if score > 0 else ('sell' if score < 0 else 'hold')

    logger.debug(
        "TrendFollowing score=%.3f direction=%s reasons=%s",
        score, direction, reasons
    )
    return score, direction


def score_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculate trend-following scores for every row in the DataFrame.
    Useful for backtesting vectorised evaluation.

    Args:
        df: Full indicator DataFrame

    Returns:
        pd.Series of float scores aligned with df index
    """
    scores = []
    for _, row in df.iterrows():
        try:
            s, _ = _score_row(row)
        except Exception:  # pylint: disable=broad-except
            s = 0.0
        scores.append(s)
    return pd.Series(scores, index=df.index, name='trend_score')
