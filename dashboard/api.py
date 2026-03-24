"""
MidasTouch - FastAPI Dashboard
Provides REST endpoints for monitoring the bot in real-time.
Runs independently of the main bot loop; reads from the shared paper_trader instance.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MidasTouch Trading Bot Dashboard",
    description="Real-time monitoring API for the MidasTouch algorithmic trading bot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
# These are injected by main.py before starting the server.
_paper_trader   = None
_data_feed      = None
_last_signals   = {}     # symbol → signal dict
_bot_running    = False


def inject_dependencies(paper_trader, data_feed, bot_running_flag: dict) -> None:
    """
    Inject shared objects from the main bot loop into the API.

    Call this from main.py before launching uvicorn.

    Args:
        paper_trader:     PaperTrader instance
        data_feed:        DataFeed instance
        bot_running_flag: Mutable dict with key 'running' (bool)
    """
    global _paper_trader, _data_feed, _bot_running
    _paper_trader = paper_trader
    _data_feed    = data_feed
    _bot_running  = bot_running_flag
    logger.info("Dashboard dependencies injected")


def update_signals(signals: dict) -> None:
    """
    Update the cached signals displayed by GET /signals.

    Args:
        signals: Dict of symbol → signal dict (from signals.generate_all_symbols)
    """
    global _last_signals
    _last_signals = signals


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/status", tags=["Bot"])
def get_status() -> dict:
    """
    Return overall bot health, capital snapshot, and drawdown.

    Returns:
        JSON with running status, capital, drawdown, and position count.
    """
    if _paper_trader is None:
        return {
            "running":         False,
            "capital":         0,
            "drawdown_pct":    0,
            "open_positions":  0,
            "message":         "Bot not initialised",
        }

    status = _paper_trader.get_status()
    running_state = _bot_running.get('running', False) if isinstance(_bot_running, dict) else _bot_running
    status['running'] = running_state
    return status


@app.get("/positions", tags=["Portfolio"])
def get_positions() -> dict:
    """
    Return all currently open positions.

    Returns:
        JSON dict of symbol → position details.
    """
    if _paper_trader is None:
        return {"positions": {}}

    positions = {}
    for symbol, pos in _paper_trader.positions.items():
        positions[symbol] = {
            "symbol":          pos.get("symbol"),
            "direction":       pos.get("direction"),
            "entry_price":     pos.get("entry_price"),
            "qty":             pos.get("qty"),
            "size_usdt":       pos.get("size_usdt"),
            "current_price":   pos.get("current_price"),
            "current_value":   pos.get("current_value"),
            "unrealised_pnl":  pos.get("unrealised_pnl"),
            "stop_loss":       pos.get("stop_loss"),
            "regime":          pos.get("regime"),
            "signal_score":    pos.get("signal_score"),
            "entry_time":      pos.get("entry_time").isoformat() if pos.get("entry_time") else None,
        }

    return {"positions": positions, "count": len(positions)}


@app.get("/trades", tags=["Portfolio"])
def get_trades(limit: int = 50) -> dict:
    """
    Return recent closed trade history.

    Args:
        limit: Maximum number of records to return (default 50, max 200)

    Returns:
        JSON with list of trade records.
    """
    if _paper_trader is None:
        return {"trades": [], "count": 0}

    limit  = min(limit, 200)
    trades = _paper_trader.get_trade_history(limit=limit)
    return {"trades": trades, "count": len(trades)}


@app.get("/performance", tags=["Analytics"])
def get_performance() -> dict:
    """
    Return full performance metrics calculated from closed trade history.

    Returns:
        JSON with Sharpe, Sortino, win rate, drawdown, P&L, and more.
    """
    if _paper_trader is None:
        return {"error": "Bot not initialised"}

    from core.performance import calculate_performance
    trades  = _paper_trader.get_trade_history(limit=10000)
    metrics = calculate_performance(trades)
    return metrics


@app.get("/signals", tags=["Trading"])
def get_signals() -> dict:
    """
    Return the most recently computed ensemble signals for each symbol.

    Returns:
        JSON dict of symbol → signal with score, direction, regime, components.
    """
    return {
        "signals": _last_signals,
        "count":   len(_last_signals),
    }


@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "MidasTouch"}


@app.get("/", tags=["System"])
def root() -> dict:
    """API root — lists available endpoints."""
    return {
        "name":      "MidasTouch Dashboard API",
        "version":   "1.0.0",
        "endpoints": [
            "GET /status",
            "GET /positions",
            "GET /trades?limit=50",
            "GET /performance",
            "GET /signals",
            "GET /health",
        ],
    }
