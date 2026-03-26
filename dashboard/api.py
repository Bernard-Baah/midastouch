"""
MidasTouch — FastAPI Dashboard
Run with: uvicorn dashboard.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import sqlite3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.performance import calculate_all

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'trades.db')
HTML_PATH = os.path.join(os.path.dirname(__file__), 'index.html')

app = FastAPI(title="MidasTouch Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bot_state = {
    'status': 'running',
    'cash': 1000.0,
    'portfolio_value': 1000.0,
    'drawdown_pct': 0.0,
    'open_positions': 0,
    'positions': {},
    'current_signals': {},
}


def update_state(state: dict):
    _bot_state.update(state)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return HTMLResponse("<h1>MidasTouch Dashboard</h1><p>UI not found.</p>")


@app.get("/status")
def get_status():
    if not os.path.exists(DB_PATH):
        return _bot_state
    conn = sqlite3.connect(DB_PATH)
    # Get open positions
    cur = conn.execute("SELECT symbol, entry_price, size_usdt, stop_loss, regime FROM trades WHERE status='open'")
    positions = {}
    for row in cur.fetchall():
        positions[row[0]] = {
            'entry_price': row[1],
            'size_usdt': row[2],
            'stop_loss': row[3],
            'regime': row[4],
        }
    # Get cash from last state (approximate)
    cur2 = conn.execute("SELECT SUM(size_usdt) FROM trades WHERE status='open'")
    open_val = cur2.fetchone()[0] or 0
    cur3 = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='closed'")
    closed_pnl = cur3.fetchone()[0] or 0
    portfolio = 1000.0 + closed_pnl
    cash = portfolio - open_val
    conn.close()
    return {
        'status': 'running',
        'cash': round(cash, 2),
        'portfolio_value': round(portfolio, 2),
        'open_positions': len(positions),
        'drawdown_pct': max(0, (1000 - portfolio) / 1000),
        'total_return_pct': round((portfolio - 1000) / 1000 * 100, 2),
    }


@app.get("/positions")
def get_positions():
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT symbol, entry_price, size_usdt, stop_loss, regime, signal_score FROM trades WHERE status='open'")
    positions = {}
    for row in cur.fetchall():
        positions[row[0]] = {
            'entry_price': row[1],
            'size_usdt': row[2],
            'stop_loss': row[3],
            'regime': row[4],
            'signal_score': row[5],
        }
    conn.close()
    return positions


@app.get("/trades")
def get_trades(limit: int = 50):
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


@app.get("/performance")
def get_performance():
    return calculate_all()


@app.get("/signals")
def get_signals():
    return _bot_state['current_signals']


@app.get("/health")
def health():
    return {"status": "ok", "bot": "running"}
