"""
MidasTouch Quant Layer — Risk Management
Volatility-based position sizing and drawdown protection.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configurable risk parameters
RISK_PER_TRADE      = 0.01   # 1% risk per trade
MIN_POSITION_USDT   = 10.0    # Never open below this
MAX_POSITION_USDT   = 30.0    # Max $30 per position
MAX_DRAWDOWN_PCT    = 0.15    # 15% portfolio drawdown kill-switch
RESERVE_PCT         = 0.20    # Keep 20% cash reserve
STOP_LOSS_VOL_MULT  = 3.5     # Was 2.0 — wider stops to avoid premature exits
TRAILING_STOP_PCT   = 0.06    # 6% trailing stop from peak price


def calculate_trailing_stop(entry_price: float, peak_price: float, direction: str = "long") -> float:
    """
    Calculate trailing stop price based on peak price seen since entry.

    For longs: stop trails below peak_price by TRAILING_STOP_PCT
    For shorts: stop trails above trough_price by TRAILING_STOP_PCT

    Args:
        entry_price: Original entry price
        peak_price:  Highest (long) or lowest (short) price seen since entry
        direction:   'long' or 'short'

    Returns:
        Trailing stop price
    """
    if direction == "long":
        trailing = peak_price * (1 - TRAILING_STOP_PCT)
        # Never move stop below entry-based initial stop
        initial_stop = entry_price * (1 - TRAILING_STOP_PCT * 1.5)
        return max(trailing, initial_stop)
    else:
        trailing = peak_price * (1 + TRAILING_STOP_PCT)
        initial_stop = entry_price * (1 + TRAILING_STOP_PCT * 1.5)
        return min(trailing, initial_stop)


def size_position(
    capital: float,
    volatility: float,
    risk_per_trade: float = RISK_PER_TRADE,
) -> float:
    """
    Volatility-adjusted position sizing.

    position_size = (capital * risk_per_trade) / volatility

    Args:
        capital:        Available capital in USDT
        volatility:     Rolling return volatility (std of pct_change)
        risk_per_trade: Fraction of capital to risk

    Returns:
        Position size in USDT, clamped to [MIN_POSITION_USDT, MAX_POSITION_USDT]
    """
    if volatility <= 0 or np.isnan(volatility):
        logger.warning("Invalid volatility %.4f — using fallback sizing", volatility)
        return min(capital * risk_per_trade, MAX_POSITION_USDT)

    # Deployable capital respects reserve
    deployable = capital * (1.0 - RESERVE_PCT)
    risk_usdt  = deployable * risk_per_trade
    size       = risk_usdt / volatility
    size       = float(np.clip(size, MIN_POSITION_USDT, MAX_POSITION_USDT))

    logger.debug(
        "Position size: $%.2f (capital=$%.2f vol=%.4f risk=%.2f%%)",
        size, capital, volatility, risk_per_trade * 100
    )
    return size


def check_drawdown(current_capital: float, peak_capital: float) -> bool:
    """
    Returns True if the portfolio has breached MAX_DRAWDOWN_PCT.
    Caller should halt trading if True.
    """
    if peak_capital <= 0:
        return False
    drawdown = (peak_capital - current_capital) / peak_capital
    if drawdown >= MAX_DRAWDOWN_PCT:
        logger.warning(
            "DRAWDOWN KILL-SWITCH: %.1f%% drawdown (peak=$%.2f current=$%.2f)",
            drawdown * 100, peak_capital, current_capital
        )
        return True
    return False


def get_volatility(df: pd.DataFrame, window: int = 14) -> float:
    """
    Compute rolling return volatility from a price DataFrame.
    Returns a single float (latest value). Safe against NaN/empty.
    """
    if df is None or df.empty or "close" not in df.columns:
        return 0.02   # fallback: 2% default vol
    vol = df["close"].pct_change().rolling(window).std().iloc[-1]
    if np.isnan(vol) or vol <= 0:
        return 0.02
    return float(vol)
