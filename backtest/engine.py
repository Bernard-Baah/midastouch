"""
MidasTouch - Walk-Forward Backtesting Engine
Evaluates strategy performance on historical data using rolling train/test windows
to avoid look-ahead bias and simulate realistic deployment.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import INITIAL_CAPITAL, PRIMARY_TIMEFRAME
from core.indicators import calculate_indicators
from core.regime_detector import detect_regime
from core.signals import generate_ensemble_signal
from core.risk_manager import RiskManager
from core.performance import calculate_performance, display_performance

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Performance metrics for a single walk-forward window."""
    window_id:        int
    train_start:      pd.Timestamp
    train_end:        pd.Timestamp
    test_start:       pd.Timestamp
    test_end:         pd.Timestamp
    trades:           List[Dict] = field(default_factory=list)
    metrics:          Dict       = field(default_factory=dict)
    final_capital:    float      = INITIAL_CAPITAL


@dataclass
class BacktestResult:
    """Aggregate results across all walk-forward windows."""
    symbol:           str
    timeframe:        str
    windows:          List[WindowResult] = field(default_factory=list)
    aggregate_metrics: Dict              = field(default_factory=dict)
    all_trades:       List[Dict]         = field(default_factory=list)


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Each window:
        • Training period  — strategy parameters are conceptually 'fitted'
          (in our case, indicators are calculated on training data to warm up)
        • Test period      — signals generated and trades simulated
    """

    def __init__(
        self,
        train_periods: int = 100,
        test_periods: int  = 50,
        step_size: int     = 25,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        """
        Args:
            train_periods:   Number of bars in the training window
            test_periods:    Number of bars in the test window
            step_size:       How many bars to advance between windows
            initial_capital: Starting capital per window
        """
        self.train_periods   = train_periods
        self.test_periods    = test_periods
        self.step_size       = step_size
        self.initial_capital = initial_capital
        self.risk_manager    = RiskManager()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = 'BTC/USDT',
        timeframe: str = PRIMARY_TIMEFRAME,
    ) -> BacktestResult:
        """
        Run walk-forward backtest on a given OHLCV DataFrame.

        Args:
            df:        OHLCV DataFrame (raw, without indicators)
            symbol:    Asset symbol for reporting
            timeframe: Candle timeframe for reporting

        Returns:
            BacktestResult with per-window and aggregate metrics
        """
        logger.info(
            "Starting backtest: %s/%s | %d bars | train=%d test=%d step=%d",
            symbol, timeframe, len(df),
            self.train_periods, self.test_periods, self.step_size,
        )

        result = BacktestResult(symbol=symbol, timeframe=timeframe)
        min_bars = self.train_periods + self.test_periods

        if len(df) < min_bars:
            logger.error(
                "Insufficient data: %d bars (need %d)", len(df), min_bars
            )
            return result

        # ── Walk-forward loop ──────────────────────────────────────────────
        window_id = 0
        start = 0

        while start + self.train_periods + self.test_periods <= len(df):
            train_end  = start + self.train_periods
            test_end   = train_end + self.test_periods

            train_df = df.iloc[start:train_end]
            test_df  = df.iloc[train_end:test_end]

            window_result = self._run_window(
                window_id=window_id,
                train_df=train_df,
                test_df=test_df,
                symbol=symbol,
            )
            result.windows.append(window_result)
            result.all_trades.extend(window_result.trades)

            start     += self.step_size
            window_id += 1

        # ── Aggregate metrics across all windows ──────────────────────────
        result.aggregate_metrics = self._aggregate(result)
        self._print_summary(result)

        return result

    # ──────────────────────────────────────────────────────────────────────────

    def _run_window(
        self,
        window_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        symbol: str,
    ) -> WindowResult:
        """
        Simulate trading on a single test window.

        Args:
            window_id: Sequential window number
            train_df:  Training period bars (used for indicator warm-up)
            test_df:   Test period bars where signals are generated
            symbol:    Asset symbol

        Returns:
            WindowResult with trades and performance metrics
        """
        window = WindowResult(
            window_id  = window_id,
            train_start = train_df.index[0],
            train_end   = train_df.index[-1],
            test_start  = test_df.index[0],
            test_end    = test_df.index[-1],
        )

        capital   = self.initial_capital
        cash      = capital
        position  = None   # Dict or None
        trades    = []

        # Combine train + running test bars for indicator calculation
        for i in range(len(test_df)):
            # Full context: train window + test bars seen so far
            context_df = pd.concat([train_df, test_df.iloc[:i + 1]])

            ind_df = calculate_indicators(context_df)
            if ind_df is None or len(ind_df) < 50:
                continue

            regime = detect_regime(ind_df)
            signal = generate_ensemble_signal(ind_df, regime)

            row         = ind_df.iloc[-1]
            close       = float(row['close'])
            atr         = float(row.get('atr_14', close * 0.02))
            current_bar = test_df.index[i]

            # ── Check stop-loss ───────────────────────────────────────────
            if position and close <= position['stop_loss']:
                pnl, cash, position = self._close_position(
                    position, close, cash, 'stop_loss'
                )
                if pnl is not None:
                    trades.append(pnl)

            # ── Signal: sell ──────────────────────────────────────────────
            elif position and signal['direction'] == 'sell' and signal['actionable']:
                pnl, cash, position = self._close_position(
                    position, close, cash, 'signal'
                )
                if pnl is not None:
                    trades.append(pnl)

            # ── Signal: buy ───────────────────────────────────────────────
            elif not position and signal['direction'] == 'buy' and signal['actionable']:
                size = self.risk_manager.calculate_atr_position_size(
                    capital=cash,
                    entry_price=close,
                    atr=atr,
                    signal_strength=abs(signal['score']),
                    regime=regime,
                )
                if size > 0:
                    stop = self.risk_manager.calculate_stop_loss(close, atr, 'buy')
                    qty  = size / close
                    cash -= size
                    position = {
                        'symbol':      symbol,
                        'entry_price': close,
                        'qty':         qty,
                        'size_usdt':   size,
                        'stop_loss':   stop,
                        'entry_time':  current_bar,
                    }

        # ── Close any remaining position at end of window ─────────────────
        if position:
            last_price = float(test_df.iloc[-1]['close'])
            pnl, cash, position = self._close_position(
                position, last_price, cash, 'end_of_window'
            )
            if pnl is not None:
                trades.append(pnl)

        window.trades        = trades
        window.final_capital = cash
        window.metrics       = calculate_performance(
            [{'pnl': t['pnl']} for t in trades],
            self.initial_capital,
        ) if trades else {}

        logger.debug(
            "Window %d: %d trades | final_capital=$%.2f",
            window_id, len(trades), cash,
        )
        return window

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        cash: float,
        reason: str,
    ) -> Tuple[Optional[Dict], float, None]:
        """
        Close a position and return the trade record + updated cash.

        Returns:
            (trade_dict, new_cash, None)
        """
        exit_value = position['qty'] * exit_price
        pnl        = exit_value - position['size_usdt']
        pnl_pct    = (pnl / position['size_usdt']) * 100

        trade = {
            'symbol':      position['symbol'],
            'entry_price': position['entry_price'],
            'exit_price':  exit_price,
            'size_usdt':   position['size_usdt'],
            'pnl':         pnl,
            'pnl_pct':     pnl_pct,
            'reason':      reason,
        }
        return trade, cash + exit_value, None

    def _aggregate(self, result: BacktestResult) -> Dict:
        """
        Compute aggregate metrics across all windows.

        Args:
            result: BacktestResult with populated windows

        Returns:
            Dict of aggregate metrics
        """
        if not result.all_trades:
            return {'error': 'No trades across all windows'}

        metrics = calculate_performance(result.all_trades, self.initial_capital)
        metrics['total_windows']     = len(result.windows)
        metrics['profitable_windows'] = sum(
            1 for w in result.windows
            if w.final_capital >= self.initial_capital
        )
        return metrics

    def _print_summary(self, result: BacktestResult) -> None:
        """Print a formatted backtest summary to the logger."""
        m = result.aggregate_metrics
        logger.info(
            "\n%s\n  Backtest: %s/%s | %d windows\n  Return: %+.2f%% | "
            "Sharpe: %.4f | WinRate: %.1f%% | MaxDD: %.2f%%\n%s",
            "=" * 60,
            result.symbol, result.timeframe, len(result.windows),
            m.get('total_return_pct', 0),
            m.get('sharpe_ratio', 0),
            m.get('win_rate', 0),
            m.get('max_drawdown_pct', 0),
            "=" * 60,
        )
