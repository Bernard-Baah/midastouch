"""
MidasTouch - Paper Trading Engine
Simulates live trading with virtual capital starting at $1,000.
All trades are logged to SQLite. Tracks cash balance, open positions, drawdown.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, Session

from config import INITIAL_CAPITAL, DATABASE_URL

logger = logging.getLogger(__name__)

Base = declarative_base()


class TradeRecord(Base):
    """SQLAlchemy model for a single trade."""

    __tablename__ = 'trades'

    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    symbol       = Column(String(20), nullable=False)
    direction    = Column(String(10), nullable=False)
    entry_price  = Column(Float, nullable=False)
    exit_price   = Column(Float, nullable=True)
    size_usdt    = Column(Float, nullable=False)
    pnl          = Column(Float, nullable=True)
    pnl_pct      = Column(Float, nullable=True)
    stop_loss    = Column(Float, nullable=True)
    reason       = Column(String(100), nullable=True)
    status       = Column(String(10), default='open')
    regime       = Column(String(20), nullable=True)
    signal_score = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<Trade id={self.id} {self.symbol} {self.direction} pnl={self.pnl}>"


class PaperTrader:
    """
    Paper trading engine with full position tracking and SQLite trade logging.

    Attributes:
        cash:         Available USDT
        positions:    Dict symbol -> open position dict
        peak_capital: Highest portfolio value seen (for drawdown)
    """

    def __init__(self, db_url: str = DATABASE_URL):
        """
        Initialise the paper trader and create/migrate the trade database.

        Args:
            db_url: SQLAlchemy database URL
        """
        if db_url.startswith('sqlite:///'):
            db_path = db_url[len('sqlite:///'):]
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.engine  = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)

        self.cash: float         = INITIAL_CAPITAL
        self.positions: Dict     = {}
        self.peak_capital: float = INITIAL_CAPITAL
        self._trade_count        = 0

        logger.info("PaperTrader ready — capital=$%.2f | db=%s", self.cash, db_url)

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def portfolio_value(self) -> float:
        """Cash + mark-to-market open positions."""
        return self.cash + sum(p['current_value'] for p in self.positions.values())

    @property
    def drawdown(self) -> float:
        """Current drawdown as fraction of peak capital."""
        pv = self.portfolio_value
        self.peak_capital = max(self.peak_capital, pv)
        return max(0.0, (self.peak_capital - pv) / self.peak_capital) if self.peak_capital > 0 else 0.0

    def update_position_prices(self, symbol: str, current_price: float) -> None:
        """Update mark-to-market value for an open position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos['current_price']  = current_price
            pos['current_value']  = pos['qty'] * current_price
            pos['unrealised_pnl'] = pos['current_value'] - pos['size_usdt']

    def execute_buy(
        self,
        symbol: str,
        size_usdt: float,
        price: float,
        stop_loss: float,
        regime: str = 'unknown',
        signal_score: float = 0.0,
        reason: str = 'signal',
    ) -> Optional[Dict]:
        """
        Execute a paper buy order.

        Args:
            symbol:       Trading pair
            size_usdt:    Position size in USDT
            price:        Entry price
            stop_loss:    Stop-loss trigger price
            regime:       Market regime at entry
            signal_score: Ensemble signal score
            reason:       Trade reason string

        Returns:
            Position dict or None if rejected
        """
        if symbol in self.positions:
            logger.warning("Already have position for %s", symbol)
            return None
        if size_usdt > self.cash:
            size_usdt = self.cash * 0.99
        if size_usdt <= 0 or price <= 0:
            return None

        qty = size_usdt / price
        self.cash -= size_usdt

        position = {
            'symbol':        symbol,
            'direction':     'buy',
            'entry_price':   price,
            'qty':           qty,
            'size_usdt':     size_usdt,
            'stop_loss':     stop_loss,
            'current_price': price,
            'current_value': size_usdt,
            'unrealised_pnl': 0.0,
            'entry_time':    datetime.now(timezone.utc),
            'regime':        regime,
            'signal_score':  signal_score,
        }
        self.positions[symbol] = position

        self._log_trade(
            symbol=symbol, direction='buy', entry_price=price,
            size_usdt=size_usdt, stop_loss=stop_loss, reason=reason,
            regime=regime, signal_score=signal_score, status='open',
        )
        self._trade_count += 1
        logger.info("BUY %s qty=%.6f @ %.4f size=$%.2f stop=%.4f cash=$%.2f",
                    symbol, qty, price, size_usdt, stop_loss, self.cash)
        return position

    def execute_sell(
        self,
        symbol: str,
        price: float,
        reason: str = 'signal',
    ) -> Optional[Dict]:
        """
        Close an open long position.

        Args:
            symbol: Trading pair
            price:  Exit price
            reason: Closure reason

        Returns:
            Trade result dict or None
        """
        if symbol not in self.positions:
            logger.warning("No position for %s", symbol)
            return None

        pos        = self.positions.pop(symbol)
        exit_value = pos['qty'] * price
        pnl        = exit_value - pos['size_usdt']
        pnl_pct    = (pnl / pos['size_usdt']) * 100 if pos['size_usdt'] > 0 else 0.0
        self.cash += exit_value

        self._close_trade(symbol=symbol, exit_price=price, pnl=pnl, pnl_pct=pnl_pct, reason=reason)

        result = {
            'symbol': symbol, 'entry_price': pos['entry_price'],
            'exit_price': price, 'size_usdt': pos['size_usdt'],
            'pnl': pnl, 'pnl_pct': pnl_pct, 'reason': reason,
        }
        logger.info("SELL %s qty=%.6f @ %.4f pnl=$%.2f (%.2f%%) cash=$%.2f reason=%s",
                    symbol, pos['qty'], price, pnl, pnl_pct, self.cash, reason)
        return result

    def check_stop_losses(self, symbol: str, current_price: float) -> bool:
        """Trigger stop-loss if price has hit the threshold."""
        if symbol not in self.positions:
            return False
        stop = self.positions[symbol].get('stop_loss')
        if stop and current_price <= stop:
            logger.warning("STOP-LOSS hit %s price=%.4f stop=%.4f", symbol, current_price, stop)
            self.execute_sell(symbol, current_price, reason='stop_loss')
            return True
        return False

    def close_all_positions(self, prices: Dict[str, float], reason: str = 'kill_switch') -> None:
        """Emergency close all open positions."""
        for symbol in list(self.positions.keys()):
            price = prices.get(symbol, self.positions[symbol]['current_price'])
            self.execute_sell(symbol, price, reason=reason)

    def get_status(self) -> Dict:
        """Return portfolio snapshot."""
        pv = self.portfolio_value
        return {
            'cash':              round(self.cash, 2),
            'portfolio_value':   round(pv, 2),
            'peak_capital':      round(self.peak_capital, 2),
            'drawdown_pct':      round(self.drawdown * 100, 2),
            'open_positions':    len(self.positions),
            'total_trades':      self._trade_count,
            'total_return_pct':  round(((pv - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2),
        }

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Return recent closed trades from the database."""
        try:
            with Session(self.engine) as session:
                records = (
                    session.query(TradeRecord)
                    .filter_by(status='closed')
                    .order_by(TradeRecord.id.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        'id':           r.id,
                        'timestamp':    r.timestamp.isoformat() if r.timestamp else None,
                        'symbol':       r.symbol,
                        'direction':    r.direction,
                        'entry_price':  r.entry_price,
                        'exit_price':   r.exit_price,
                        'size_usdt':    r.size_usdt,
                        'pnl':          r.pnl,
                        'pnl_pct':      r.pnl_pct,
                        'reason':       r.reason,
                        'regime':       r.regime,
                        'signal_score': r.signal_score,
                    }
                    for r in records
                ]
        except Exception as exc:
            logger.error("Failed to fetch trade history: %s", exc)
            return []

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _log_trade(self, **kwargs) -> None:
        """Insert a new trade record."""
        try:
            with Session(self.engine) as session:
                session.add(TradeRecord(**kwargs))
                session.commit()
        except Exception as exc:
            logger.error("Failed to log trade: %s", exc)

    def _close_trade(self, symbol: str, exit_price: float,
                     pnl: float, pnl_pct: float, reason: str) -> None:
        """Update the open trade record with exit data."""
        try:
            with Session(self.engine) as session:
                record = (
                    session.query(TradeRecord)
                    .filter_by(symbol=symbol, status='open')
                    .order_by(TradeRecord.id.desc())
                    .first()
                )
                if record:
                    record.exit_price = exit_price
                    record.pnl        = pnl
                    record.pnl_pct    = pnl_pct
                    record.reason     = reason
                    record.status     = 'closed'
                    session.commit()
        except Exception as exc:
            logger.error("Failed to close trade record: %s", exc)
