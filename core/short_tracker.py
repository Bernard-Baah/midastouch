"""
MidasTouch — Short Position Tracker (Paper Mode)
Simulates short selling without real margin.
A short is: sell at entry price, profit when price drops.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ShortTracker:
    """Tracks simulated short positions in paper trading mode."""

    def __init__(self):
        self.shorts: dict = {}   # symbol -> {entry_price, size_usdt, stop_loss}
        self.closed: list = []

    def open_short(
        self,
        symbol: str,
        entry_price: float,
        size_usdt: float,
        stop_loss: float,
        reason: str = "quant_short",
    ) -> bool:
        """Open a simulated short position."""
        if symbol in self.shorts:
            logger.warning("Already short %s — skipping", symbol)
            return False

        self.shorts[symbol] = {
            "entry_price": entry_price,
            "size_usdt":   size_usdt,
            "stop_loss":   stop_loss,
            "qty":         size_usdt / entry_price,
            "open_time":   datetime.now(timezone.utc).isoformat(),
            "reason":      reason,
        }
        logger.info("SHORT OPEN  %s @ %.4f | size=$%.2f | stop=%.4f", symbol, entry_price, size_usdt, stop_loss)
        return True

    def close_short(self, symbol: str, exit_price: float, reason: str = "signal") -> Optional[dict]:
        """Close a short position and calculate P&L."""
        pos = self.shorts.pop(symbol, None)
        if pos is None:
            return None

        # Short P&L: profit when price falls
        pnl     = (pos["entry_price"] - exit_price) * pos["qty"]
        pnl_pct = pnl / pos["size_usdt"] * 100

        result = {
            "symbol":      symbol,
            "entry_price": pos["entry_price"],
            "exit_price":  exit_price,
            "size_usdt":   pos["size_usdt"],
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "reason":      reason,
        }
        self.closed.append(result)

        emoji = "✅" if pnl >= 0 else "❌"
        logger.info("SHORT CLOSE %s @ %.4f | PnL=$%.2f (%.1f%%) %s", symbol, exit_price, pnl, pnl_pct, emoji)
        return result

    def check_stop_losses(self, symbol: str, current_price: float) -> bool:
        """Trigger stop loss if price rises above stop level."""
        pos = self.shorts.get(symbol)
        if pos and current_price >= pos["stop_loss"]:
            self.close_short(symbol, current_price, reason="stop_loss")
            return True
        return False

    def get_open_shorts(self) -> dict:
        return dict(self.shorts)

    def get_unrealized_pnl(self, current_prices: dict) -> float:
        """Calculate unrealized P&L across all open shorts."""
        total = 0.0
        for sym, pos in self.shorts.items():
            price = current_prices.get(sym, pos["entry_price"])
            pnl   = (pos["entry_price"] - price) * pos["qty"]
            total += pnl
        return total
