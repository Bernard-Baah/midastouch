"""
MidasTouch — Performance Tracker
Calculates Sharpe, Sortino, win rate, profit factor, max drawdown.
"""

import numpy as np
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'trades.db')


def get_closed_trades():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT pnl, pnl_pct FROM trades WHERE status='closed'")
    return cur.fetchall()


def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free / periods_per_year
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(returns, risk_free=0.0, periods_per_year=252):
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('inf')
    downside_std = np.std(downside)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve):
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 2)


def calculate_all(paper_trader=None):
    trades = get_closed_trades()
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe': 0,
            'sortino': 0,
            'max_drawdown_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'best_trade': 0,
            'worst_trade': 0,
        }

    pnls = [t[0] for t in trades]
    pct_returns = [t[1] / 100 for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_trades': len(pnls),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe_ratio(pct_returns), 2),
        'sortino': round(sortino_ratio(pct_returns), 2),
        'avg_win': round(np.mean(wins), 2) if wins else 0,
        'avg_loss': round(np.mean(losses), 2) if losses else 0,
        'best_trade': round(max(pnls), 2) if pnls else 0,
        'worst_trade': round(min(pnls), 2) if pnls else 0,
        'total_pnl': round(sum(pnls), 2),
    }
