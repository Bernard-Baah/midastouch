"""
MidasTouch - Risk Manager
Handles position sizing, stop-loss calculation, and drawdown kill-switch.
All sizing uses ATR-based risk to normalise exposure across different volatility regimes.
"""

import logging
from typing import Optional

from config import (
    MIN_RISK_PER_TRADE, MAX_RISK_PER_TRADE,
    MAX_DRAWDOWN_LIMIT, RESERVE_CAPITAL_PCT,
    MAX_CAPITAL_PER_TRADE, STOP_LOSS_ATR_MULT
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralised risk calculation for the MidasTouch trading bot.
    """

    # Regime multipliers on base risk percentage
    REGIME_RISK_MULT = {
        'bull':            1.00,
        'bear':            0.60,
        'sideways':        0.75,
        'high_volatility': 0.50,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Public Methods
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        capital: float,
        atr: float,
        signal_strength: float,
        regime: str = 'sideways',
    ) -> float:
        """
        Calculate position size in USDT using ATR-based risk sizing.

        Formula:
            base_risk_pct  = lerp(MIN_RISK, MAX_RISK, signal_strength)
            regime_adj_pct = base_risk_pct * regime_multiplier
            risk_amount    = capital * regime_adj_pct
            atr_units      = risk_amount / atr           (# units to risk)
            position_usdt  = atr_units * current_price
               → simplified to: risk_amount / atr_pct_of_price
               (we use risk_amount as the raw USDT exposure since ATR is in price units)

        The final size is capped at MAX_CAPITAL_PER_TRADE of available capital.

        Args:
            capital:         Available capital in USDT
            atr:             ATR value in price units (must be > 0)
            signal_strength: Absolute ensemble score (0.0 – 1.0)
            regime:          Current market regime string

        Returns:
            Position size in USDT (minimum 0)
        """
        if capital <= 0 or atr <= 0:
            logger.warning("Invalid capital=%.2f or atr=%.4f for sizing", capital, atr)
            return 0.0

        signal_strength = max(0.0, min(1.0, abs(signal_strength)))

        # Interpolate risk% between min and max based on signal strength
        base_risk_pct = MIN_RISK_PER_TRADE + (
            (MAX_RISK_PER_TRADE - MIN_RISK_PER_TRADE) * signal_strength
        )

        # Apply regime multiplier
        regime_mult = self.REGIME_RISK_MULT.get(regime, 0.75)
        adj_risk_pct = base_risk_pct * regime_mult
        adj_risk_pct = max(MIN_RISK_PER_TRADE, min(MAX_RISK_PER_TRADE, adj_risk_pct))

        risk_amount = capital * adj_risk_pct

        # ATR-based sizing: number of price-units we can afford to risk
        # position_size = risk_amount / (atr per unit) — but since we're working
        # in USDT directly we treat risk_amount / (atr / price) i.e. scale by price
        # Simplified: position_usdt = risk_amount * (price / atr)
        # We don't have price here, so return risk_amount as the USDT position
        # and let the caller divide by price for unit count.
        # For paper trading purposes: size = risk_amount / atr * entry_price
        # We return the USDT value of the position.
        position_usdt = risk_amount  # caller multiplies by price/atr ratio

        # Hard cap at MAX_CAPITAL_PER_TRADE of available capital
        max_allowed = capital * MAX_CAPITAL_PER_TRADE
        position_usdt = min(position_usdt, max_allowed)

        logger.debug(
            "Position size: capital=%.2f atr=%.4f signal=%.3f regime=%s "
            "risk_pct=%.3f position=%.2f USDT",
            capital, atr, signal_strength, regime, adj_risk_pct, position_usdt,
        )
        return round(position_usdt, 2)

    def calculate_atr_position_size(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        signal_strength: float,
        regime: str = 'sideways',
    ) -> float:
        """
        Full ATR-based position sizing in USDT.

        position_usdt = (capital * risk_pct) / (atr / entry_price)

        Args:
            capital:         Available capital in USDT
            entry_price:     Asset entry price
            atr:             ATR value in price units
            signal_strength: Absolute signal score 0-1
            regime:          Market regime string

        Returns:
            Position size in USDT
        """
        if entry_price <= 0 or atr <= 0 or capital <= 0:
            return 0.0

        signal_strength = max(0.0, min(1.0, abs(signal_strength)))
        base_risk_pct = MIN_RISK_PER_TRADE + (
            (MAX_RISK_PER_TRADE - MIN_RISK_PER_TRADE) * signal_strength
        )
        regime_mult  = self.REGIME_RISK_MULT.get(regime, 0.75)
        adj_risk_pct = max(MIN_RISK_PER_TRADE, min(MAX_RISK_PER_TRADE, base_risk_pct * regime_mult))

        risk_amount  = capital * adj_risk_pct
        atr_pct      = atr / entry_price              # ATR as fraction of price
        position_usdt = risk_amount / atr_pct          # USDT exposure

        max_allowed   = capital * MAX_CAPITAL_PER_TRADE
        position_usdt = min(position_usdt, max_allowed)

        logger.debug(
            "ATR Position: %.2f USDT | risk_pct=%.4f entry=%.4f atr=%.4f",
            position_usdt, adj_risk_pct, entry_price, atr,
        )
        return round(position_usdt, 2)

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = 'buy',
    ) -> float:
        """
        Calculate stop-loss price at STOP_LOSS_ATR_MULT × ATR from entry.

        Args:
            entry_price: Trade entry price
            atr:         ATR value in price units
            direction:   'buy' (long) → stop below entry
                         'sell' (short) → stop above entry

        Returns:
            Stop-loss price
        """
        if entry_price <= 0 or atr <= 0:
            logger.warning("Invalid entry=%.4f or atr=%.4f for stop-loss", entry_price, atr)
            return entry_price

        offset = STOP_LOSS_ATR_MULT * atr
        stop = entry_price - offset if direction == 'buy' else entry_price + offset

        logger.debug(
            "Stop-loss: entry=%.4f atr=%.4f direction=%s stop=%.4f",
            entry_price, atr, direction, stop,
        )
        return round(stop, 8)

    def check_drawdown(
        self,
        current_capital: float,
        peak_capital: float,
    ) -> bool:
        """
        Determine whether the drawdown kill-switch should be triggered.

        Args:
            current_capital: Current portfolio value
            peak_capital:    Highest portfolio value seen

        Returns:
            True if drawdown exceeds MAX_DRAWDOWN_LIMIT (bot should halt)
        """
        if peak_capital <= 0:
            return False

        drawdown = (peak_capital - current_capital) / peak_capital
        triggered = drawdown >= MAX_DRAWDOWN_LIMIT

        if triggered:
            logger.warning(
                "KILL-SWITCH TRIGGERED: drawdown=%.2f%% (limit=%.2f%%)",
                drawdown * 100, MAX_DRAWDOWN_LIMIT * 100,
            )
        else:
            logger.debug(
                "Drawdown check: %.2f%% / limit %.2f%%",
                drawdown * 100, MAX_DRAWDOWN_LIMIT * 100,
            )
        return triggered

    def get_current_drawdown(
        self,
        current_capital: float,
        peak_capital: float,
    ) -> float:
        """
        Return current drawdown as a decimal fraction.

        Args:
            current_capital: Current portfolio value
            peak_capital:    Peak portfolio value

        Returns:
            Drawdown fraction (0.0 = no drawdown, 0.15 = 15% drawdown)
        """
        if peak_capital <= 0:
            return 0.0
        return max(0.0, (peak_capital - current_capital) / peak_capital)
