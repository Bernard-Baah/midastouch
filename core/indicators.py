"""
MidasTouch - Technical Indicators Module
Calculates all required indicators and appends them as columns to a DataFrame.
Uses the `ta` library for robust, battle-tested implementations.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import ta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from config import (
    EMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD, VOLUME_SMA_PERIOD, ATR_AVG_PERIOD
)

logger = logging.getLogger(__name__)


def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculate all technical indicators and append as new columns.

    Adds the following columns to the returned DataFrame:
        ema_9, ema_21, ema_50, ema_200
        rsi_14
        macd_line, macd_signal, macd_hist
        bb_upper, bb_middle, bb_lower, bb_pct_b
        atr_14
        atr_avg_30          (rolling 30-period average of ATR)
        volume_sma_20

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with indicator columns appended, or None on failure.
    """
    if df is None or df.empty:
        logger.warning("calculate_indicators received empty DataFrame")
        return None

    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.error("Missing columns for indicators: %s", missing)
        return None

    out = df.copy()

    try:
        # ── EMAs ─────────────────────────────────────────────────────────────
        for period in EMA_PERIODS:
            ema = EMAIndicator(close=out['close'], window=period, fillna=True)
            out[f'ema_{period}'] = ema.ema_indicator()

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi = RSIIndicator(close=out['close'], window=RSI_PERIOD, fillna=True)
        out['rsi_14'] = rsi.rsi()

        # ── MACD ─────────────────────────────────────────────────────────────
        macd = MACD(
            close=out['close'],
            window_fast=MACD_FAST,
            window_slow=MACD_SLOW,
            window_sign=MACD_SIGNAL,
            fillna=True,
        )
        out['macd_line']   = macd.macd()
        out['macd_signal'] = macd.macd_signal()
        out['macd_hist']   = macd.macd_diff()

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb = BollingerBands(
            close=out['close'],
            window=BB_PERIOD,
            window_dev=BB_STD,
            fillna=True,
        )
        out['bb_upper']  = bb.bollinger_hband()
        out['bb_middle'] = bb.bollinger_mavg()
        out['bb_lower']  = bb.bollinger_lband()
        out['bb_pct_b']  = bb.bollinger_pband()

        # ── ATR ──────────────────────────────────────────────────────────────
        atr = AverageTrueRange(
            high=out['high'],
            low=out['low'],
            close=out['close'],
            window=ATR_PERIOD,
            fillna=True,
        )
        out['atr_14'] = atr.average_true_range()

        # Rolling average of ATR (for regime detection)
        out['atr_avg_30'] = out['atr_14'].rolling(window=ATR_AVG_PERIOD, min_periods=1).mean()

        # ── Volume SMA ───────────────────────────────────────────────────────
        out['volume_sma_20'] = (
            out['volume'].rolling(window=VOLUME_SMA_PERIOD, min_periods=1).mean()
        )

        logger.debug("Indicators calculated — %d rows", len(out))
        return out

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error calculating indicators: %s", exc, exc_info=True)
        return None


def get_latest_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return the most recent row of a DataFrame as a Series.

    Args:
        df: DataFrame with indicator columns

    Returns:
        pd.Series of the last row, or None if df is empty/None
    """
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def validate_indicators(row: pd.Series) -> bool:
    """
    Check that all required indicator values are present and non-NaN.

    Args:
        row: A single row from an indicators DataFrame

    Returns:
        True if all expected indicator columns are valid
    """
    required_cols = [
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'rsi_14',
        'macd_line', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr_14', 'atr_avg_30',
        'volume_sma_20',
    ]
    for col in required_cols:
        if col not in row.index or pd.isna(row[col]):
            logger.debug("Missing/NaN indicator: %s", col)
            return False
    return True
