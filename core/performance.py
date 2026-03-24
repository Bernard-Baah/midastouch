"""
MidasTouch - Performance Analytics
Calculates Sharpe, Sortino, win rate, profit factor, max drawdown and more
from the trade history stored in the paper trader.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from config import INITIAL_CAPITAL

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 365  # crypto never sleeps
RISK_FREE_RATE        = 0.05  # 5% annual


def calculate_performance(trades: List[Dict], initial_capital: float = INITIAL_CAPITAL) -> Dict:
    """
    Calculate a full suite of performance metrics from closed trade history.

    Args:
        trades:          List of trade dicts (with at least a 'pnl' key)
        initial_capital: Starting capital

    Returns:
        Dict of metric name -> value
    """
    if not trades:
        return _empty_metrics(initial_capital)

    pnl_list = [t['pnl'] for t in trades if t.get('pnl') is not None]
    if not pnl_list:
        return _empty_metrics(initial_capital)

    pnl_arr     = np.array(pnl_list, dtype=float)
    returns_arr = pnl_arr / initial_capital

    total_pnl        = float(np.sum(pnl_arr))
    total_return_pct = (total_pnl / initial_capital) * 100.0

    wins     = pnl_arr[pnl_arr > 0]
    losses   = pnl_arr[pnl_arr < 0]
    n_total  = len(pnl_arr)
    n_wins   = len(wins)
    n_losses = len(losses)

    win_rate    = n_wins / n_total if n_total else 0.0
    avg_win     = float(np.mean(wins))   if n_wins   else 0.0
    avg_loss    = float(np.mean(losses)) if n_losses else 0.0
    best_trade  = float(np.max(pnl_arr))
    worst_trade = float(np.min(pnl_arr))

    gross_profit  = float(np.sum(wins))           if n_wins   else 0.0
    gross_loss    = abs(float(np.sum(losses)))     if n_losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    sharpe  = _sharpe_ratio(returns_arr)
    sortino = _sortino_ratio(returns_arr)

    equity_curve = _build_equity_curve(pnl_arr, initial_capital)
    max_dd       = _max_drawdown(equity_curve)

    return {
        'total_pnl':          round(total_pnl, 2),
        'total_return_pct':   round(total_return_pct, 2),
        'total_trades':       n_total,
        'winning_trades':     n_wins,
        'losing_trades':      n_losses,
        'win_rate':           round(win_rate * 100, 2),
        'profit_factor':      round(profit_factor, 3),
        'avg_win_usdt':       round(avg_win, 2),
        'avg_loss_usdt':      round(avg_loss, 2),
        'best_trade_usdt':    round(best_trade, 2),
        'worst_trade_usdt':   round(worst_trade, 2),
        'sharpe_ratio':       round(sharpe, 4),
        'sortino_ratio':      round(sortino, 4),
        'max_drawdown_pct':   round(max_dd * 100, 2),
        'equity_curve':       [round(v, 2) for v in equity_curve],
    }


def display_performance(metrics: Dict) -> str:
    """
    Format performance metrics as a human-readable string.

    Args:
        metrics: Dict from calculate_performance()

    Returns:
        Formatted multi-line string
    """
    lines = [
        "=" * 50,
        "  MidasTouch Performance Report",
        "=" * 50,
        f"  Total Return:      {metrics.get('total_return_pct', 0):+.2f}%  (${metrics.get('total_pnl', 0):+.2f})",
        f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.4f}",
        f"  Sortino Ratio:     {metrics.get('sortino_ratio', 0):.4f}",
        f"  Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"  Win Rate:          {metrics.get('win_rate', 0):.1f}%",
        f"  Profit Factor:     {metrics.get('profit_factor', 0):.3f}",
        f"  Total Trades:      {metrics.get('total_trades', 0)}",
        f"  Wins / Losses:     {metrics.get('winning_trades', 0)} / {metrics.get('losing_trades', 0)}",
        f"  Best Trade:        ${metrics.get('best_trade_usdt', 0):+.2f}",
        f"  Worst Trade:       ${metrics.get('worst_trade_usdt', 0):+.2f}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sharpe_ratio(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess   = returns - daily_rf
    std      = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float((np.mean(excess) / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino_ratio(returns: np.ndarray) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    daily_rf    = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess      = returns - daily_rf
    downside    = excess[excess < 0]
    if len(downside) == 0:
        return float('inf')
    down_std = np.std(downside, ddof=1)
    if down_std == 0:
        return 0.0
    return float((np.mean(excess) / down_std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _build_equity_curve(pnl_arr: np.ndarray, initial_capital: float) -> List[float]:
    """Build cumulative equity curve from trade P&L array."""
    curve   = [initial_capital]
    running = initial_capital
    for pnl in pnl_arr:
        running += pnl
        curve.append(running)
    return curve


def _max_drawdown(equity_curve: List[float]) -> float:
    """Maximum drawdown as a fraction from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    arr  = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (peak - arr) / np.where(peak > 0, peak, 1)
    return float(np.max(dd))


def _empty_metrics(initial_capital: float) -> Dict:
    """Zeroed metrics dict when there are no trades."""
    return {
        'total_pnl':          0.0,
        'total_return_pct':   0.0,
        'total_trades':       0,
        'winning_trades':     0,
        'losing_trades':      0,
        'win_rate':           0.0,
        'profit_factor':      0.0,
        'avg_win_usdt':       0.0,
        'avg_loss_usdt':      0.0,
        'best_trade_usdt':    0.0,
        'worst_trade_usdt':   0.0,
        'sharpe_ratio':       0.0,
        'sortino_ratio':      0.0,
        'max_drawdown_pct':   0.0,
        'equity_curve':       [initial_capital],
    }

# Alias for compatibility
def calculate_all():
    from core.paper_trader import PaperTrader
    import sqlite3, os
    db = os.path.join(os.path.dirname(__file__), '..', 'data', 'trades.db')
    if not os.path.exists(db):
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'sortino': 0, 'total_pnl': 0}
    conn = sqlite3.connect(db)
    cur = conn.execute("SELECT pnl FROM trades WHERE status='closed'")
    pnls = [r[0] for r in cur.fetchall()]
    if not pnls:
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'sortino': 0, 'total_pnl': 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    profit_factor = sum(wins) / abs(sum(losses)) if losses else 0
    return {
        'total_trades': len(pnls),
        'win_rate': round(len(wins)/len(pnls)*100, 1),
        'profit_factor': round(profit_factor, 2),
        'total_pnl': round(sum(pnls), 2),
        'sharpe': 0, 'sortino': 0,
    }

