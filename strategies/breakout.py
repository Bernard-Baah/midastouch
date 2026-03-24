"""
MidasTouch - Breakout Strategy
Volume-confirmed price breakout signals.

Signal Logic:
    BUY  when: price breaks above 20-period high with volume >= 1.5x average
    SELL when: price breaks below 20-period low
    Score range: -1.0 (strong sell) to +1.0 (strong buy)
"""

import logging
from typing import Tuple

import pandas as pd

from config import BREAKOUT_LOOKBACK, BREAKOUT_VOLUME_MULT
from core.indicators import get_latest_row, validate_indicators

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Generate a breakout signal from the latest bars.

    Requires at least BREAKOUT_LOOKBACK + 1 rows to calculate rolling
    high/low correctly.

    Score components:
        +0.60  price above rolling 20-period high (base breakout)
        +0.40  volume >= 1.5x average (confirmation)
        -0.70  price below rolling 20-period low (breakdown)

    Args:
        df: DataFrame with indicator columns already calculated.

    Returns:
        Tuple of (score: float, reason: str)
    """
    if df is None or len(df) < BREAKOUT_LOOKBACK + 1:
        logger.debug("Breakout: insufficient data")
        return 0.0, 'hold'

    row = get_latest_row(df)
    if row is None or not validate_indicators(row):
        logger.debug("Breakout: invalid indicators")
        return 0.0, 'hold'

    return _score_row(df, row)


def _score_row(df: pd.DataFrame, row: pd.Series) -> Tuple[float, str]:
    """
    Calculate breakout score using rolling high/low from df history.

    We use the PREVIOUS bar's rolling high/low (excluding the current bar)
    to avoid look-ahead bias.

    Args:
        df:  Full indicator DataFrame
        row: Latest row (pd.Series)

    Returns:
        (score, direction_string)
    """
    # Lookback window excludes current bar
    lookback_df = df.iloc[-(BREAKOUT_LOOKBACK + 1):-1]

    rolling_high = float(lookback_df['high'].max())
    rolling_low  = float(lookback_df['low'].min())

    close  = float(row['close'])
    volume = float(row['volume'])
    vol_avg = float(row['volume_sma_20'])

    score = 0.0
    reasons = []

    # ── Upside breakout ───────────────────────────────────────────────────────
    if close > rolling_high:
        score += 0.60
        reasons.append(f'break_above_{rolling_high:.2f}')

        if vol_avg > 0 and volume >= vol_avg * BREAKOUT_VOLUME_MULT:
            score += 0.40
            reasons.append('volume_confirmed')
        else:
            # Unconfirmed breakout — weaker signal
            score -= 0.20
            reasons.append('low_volume_breakout')

    # ── Downside breakdown ────────────────────────────────────────────────────
    elif close < rolling_low:
        score -= 0.70
        reasons.append(f'break_below_{rolling_low:.2f}')

        if vol_avg > 0 and volume >= vol_avg * BREAKOUT_VOLUME_MULT:
            score -= 0.30
            reasons.append('breakdown_volume_confirmed')

    score = max(-1.0, min(1.0, score))
    direction = 'buy' if score > 0 else ('sell' if score < 0 else 'hold')

    logger.debug(
        "Breakout score=%.3f direction=%s rolling_high=%.4f rolling_low=%.4f reasons=%s",
        score, direction, rolling_high, rolling_low, reasons
    )
    return score, direction


def score_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculate breakout scores for every eligible row — useful for backtesting.

    Args:
        df: Full indicator DataFrame

    Returns:
        pd.Series of float scores
    """
    scores = []
    for i in range(len(df)):
        if i < BREAKOUT_LOOKBACK:
            scores.append(0.0)
            continue
        sub_df = df.iloc[:i + 1]
        row = sub_df.iloc[-1]
        try:
            s, _ = _score_row(sub_df, row)
        except Exception:  # pylint: disable=broad-except
            s = 0.0
        scores.append(s)
    return pd.Series(scores, index=df.index, name='breakout_score')
