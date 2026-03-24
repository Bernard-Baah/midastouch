"""
MidasTouch — Paper Trading Engine
Simulates trades with $1,000 virtual capital. Logs all trades to SQLite.
"""

import sqlite3
import os
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'trades.db')


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            size_usdt REAL,
            stop_loss REAL,
            pnl REAL,
            pnl_pct REAL,
            reason TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    return conn


class PaperTrader:
    def __init__(self, initial_capital=1000.0):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.positions = {}  # symbol -> {size, entry_price, stop_loss, size_usdt}
        self.trade_count = 0
        self.conn = _get_conn()

    @property
    def total_capital(self):
        """Cash + value of open positions."""
        pos_value = sum(
            p['size'] * p['entry_price'] for p in self.positions.values()
        )
        return self.cash + pos_value

    @property
    def drawdown(self):
        total = self.total_capital
        if total > self.peak_capital:
            self.peak_capital = total
        return (self.peak_capital - total) / self.peak_capital

    def execute_buy(self, symbol, size_usdt, price, stop_loss, reason='signal'):
        """Open a long position."""
        if symbol in self.positions:
            return False, 'Already in position'
        if size_usdt > self.cash:
            return False, 'Insufficient cash'

        units = size_usdt / price
        self.cash -= size_usdt
        self.positions[symbol] = {
            'size': units,
            'entry_price': price,
            'stop_loss': stop_loss,
            'size_usdt': size_usdt,
            'open_time': datetime.utcnow().isoformat(),
        }

        self.conn.execute('''
            INSERT INTO trades (timestamp, symbol, direction, entry_price, size_usdt, stop_loss, reason, status)
            VALUES (?, ?, 'buy', ?, ?, ?, ?, 'open')
        ''', (datetime.utcnow().isoformat(), symbol, price, size_usdt, stop_loss, reason))
        self.conn.commit()
        self.trade_count += 1
        print(f"  [PAPER BUY]  {symbol} @ {price:.4f} | Size: ${size_usdt:.2f} | SL: {stop_loss:.4f}")
        return True, 'OK'

    def execute_sell(self, symbol, price, reason='signal'):
        """Close a long position."""
        if symbol not in self.positions:
            return False, 'No open position'

        pos = self.positions.pop(symbol)
        proceeds = pos['size'] * price
        pnl = proceeds - pos['size_usdt']
        pnl_pct = pnl / pos['size_usdt'] * 100
        self.cash += proceeds

        self.conn.execute('''
            UPDATE trades SET exit_price=?, pnl=?, pnl_pct=?, status='closed', reason=?
            WHERE symbol=? AND status='open'
        ''', (price, pnl, pnl_pct, reason, symbol))
        self.conn.commit()

        emoji = '✅' if pnl >= 0 else '❌'
        print(f"  [PAPER SELL] {symbol} @ {price:.4f} | PnL: ${pnl:.2f} ({pnl_pct:.1f}%) {emoji}")
        return True, pnl

    def check_stop_losses(self, current_prices):
        """Check and trigger stop losses for open positions."""
        for symbol, pos in list(self.positions.items()):
            price = current_prices.get(symbol)
            if price and price <= pos['stop_loss']:
                self.execute_sell(symbol, price, reason='stop_loss')

    def get_open_positions(self):
        return dict(self.positions)

    def get_trade_history(self, limit=50):
        cur = self.conn.execute(
            'SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self):
        total = self.total_capital
        return {
            'cash': round(self.cash, 2),
            'total_capital': round(total, 2),
            'initial_capital': self.initial_capital,
            'total_return_pct': round((total - self.initial_capital) / self.initial_capital * 100, 2),
            'drawdown_pct': round(self.drawdown * 100, 2),
            'open_positions': len(self.positions),
            'total_trades': self.trade_count,
        }
