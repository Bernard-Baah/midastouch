"""
MidasTouch - Main Bot Loop
Orchestrates data fetching, indicator calculation, signal generation,
risk management, and paper trade execution on a 5-minute cycle.

Usage:
    python main.py              # Run the bot (paper trading)
    python main.py --dashboard  # Run bot + FastAPI dashboard on :8000
    python main.py --backtest   # Run a quick backtest on BTC/USDT
"""

import argparse
import logging
import sys
import time
import threading
from datetime import datetime, timezone

import uvicorn

from config import (
    SYMBOLS, PRIMARY_TIMEFRAME, LOOP_INTERVAL_SECONDS,
    INITIAL_CAPITAL, LOG_LEVEL, LOG_FORMAT
)
from core.data_feed      import DataFeed
from core.indicators     import calculate_indicators
from core.regime_detector import detect_regime, regime_summary
from core.signals        import generate_all_symbols
from core.risk_manager   import RiskManager
from core.paper_trader   import PaperTrader
from core.performance    import calculate_performance, display_performance
from dashboard.api       import app as dashboard_app, inject_dependencies, update_signals

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger('midastouch.main')


# ─── Bot ──────────────────────────────────────────────────────────────────────

class MidasTouchBot:
    """
    The main trading bot controller.
    Wires together all components and drives the event loop.
    """

    def __init__(self):
        logger.info("🟡 Initialising MidasTouch...")
        self.data_feed    = DataFeed()
        self.risk_manager = RiskManager()
        self.paper_trader = PaperTrader()
        self.running_flag = {'running': False}
        self._kill_switch_active = False

        logger.info(
            "✅ MidasTouch ready | capital=$%.2f | symbols=%s",
            self.paper_trader.cash, SYMBOLS
        )

    # ──────────────────────────────────────────────────────────────────────────

    def run_loop(self) -> None:
        """
        Main trading loop. Runs indefinitely until kill-switch triggers
        or a KeyboardInterrupt is received.
        """
        self.running_flag['running'] = True
        logger.info("🚀 Bot loop started — interval: %ds", LOOP_INTERVAL_SECONDS)

        while self.running_flag['running']:
            loop_start = time.time()

            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("⛔ KeyboardInterrupt — stopping bot")
                self.running_flag['running'] = False
                break
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Unhandled error in tick: %s", exc, exc_info=True)

            # Respect loop interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed)
            logger.info(
                "⏱  Loop complete in %.1fs | sleeping %.0fs...", elapsed, sleep_time
            )
            if sleep_time > 0 and self.running_flag['running']:
                time.sleep(sleep_time)

        logger.info("Bot loop exited.")

    def _tick(self) -> None:
        """Execute one full iteration of the trading loop."""
        now = datetime.now(timezone.utc)
        logger.info("=" * 60)
        logger.info("🕐 Tick @ %s", now.strftime('%Y-%m-%d %H:%M:%S UTC'))
        logger.info("=" * 60)

        # ── Kill-switch check ─────────────────────────────────────────────
        if self._kill_switch_active:
            logger.warning("⛔ KILL-SWITCH ACTIVE — no new trades")
            self._print_status()
            return

        # ── Check drawdown kill-switch ────────────────────────────────────
        pv = self.paper_trader.portfolio_value
        dd = self.paper_trader.drawdown
        if self.risk_manager.check_drawdown(pv, self.paper_trader.peak_capital):
            self._kill_switch_active = True
            logger.critical(
                "🚨 DRAWDOWN KILL-SWITCH TRIGGERED — closing all positions"
            )
            # Fetch current prices for all open positions
            prices = {}
            for sym in list(self.paper_trader.positions.keys()):
                price = self.data_feed.get_current_price(sym)
                if price:
                    prices[sym] = price
            self.paper_trader.close_all_positions(prices, reason='kill_switch')
            self.running_flag['running'] = False
            return

        # ── Fetch data and compute indicators per symbol ──────────────────
        indicator_data = {}   # symbol → indicator DataFrame
        symbol_regimes = {}   # symbol → regime string

        for symbol in SYMBOLS:
            df = self.data_feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME)
            if df is None or df.empty:
                logger.warning("No data for %s — skipping", symbol)
                continue

            ind_df = calculate_indicators(df)
            if ind_df is None:
                logger.warning("Indicator calculation failed for %s", symbol)
                continue

            regime = detect_regime(ind_df)
            indicator_data[symbol] = ind_df
            symbol_regimes[symbol] = regime

            # Update position mark-to-market
            current_price = float(ind_df.iloc[-1]['close'])
            self.paper_trader.update_position_prices(symbol, current_price)

            # Check stop-losses
            self.paper_trader.check_stop_losses(symbol, current_price)

            r_sum = regime_summary(ind_df)
            logger.info(
                "  %s | regime=%s | price=%.4f | rsi=%.1f | atr=%.4f",
                symbol.ljust(10), regime.ljust(16),
                r_sum.get('close', 0), r_sum.get('rsi_14', 0), r_sum.get('atr_14', 0)
            )

        if not indicator_data:
            logger.warning("No valid data for any symbol this tick")
            self._print_status()
            return

        # ── Generate ensemble signals ─────────────────────────────────────
        signals = generate_all_symbols(indicator_data, symbol_regimes)
        update_signals(signals)  # push to dashboard

        # ── Execute trades based on signals ───────────────────────────────
        for symbol, signal in signals.items():
            self._process_signal(symbol, signal, indicator_data[symbol])

        self._print_status()

    def _process_signal(
        self,
        symbol: str,
        signal: dict,
        ind_df,
    ) -> None:
        """
        Act on an ensemble signal — buy, sell, or hold.

        Args:
            symbol: Trading pair
            signal: Signal dict from generate_ensemble_signal
            ind_df: Indicator DataFrame for the symbol
        """
        direction  = signal.get('direction', 'hold')
        score      = signal.get('score', 0.0)
        actionable = signal.get('actionable', False)
        regime     = signal.get('regime', 'sideways')

        if not actionable:
            logger.info("  %s → HOLD (score=%.3f)", symbol, score)
            return

        row   = ind_df.iloc[-1]
        price = float(row['close'])
        atr   = float(row.get('atr_14', price * 0.02))

        # ── BUY ──────────────────────────────────────────────────────────
        if direction == 'buy' and symbol not in self.paper_trader.positions:
            size = self.risk_manager.calculate_atr_position_size(
                capital         = self.paper_trader.cash,
                entry_price     = price,
                atr             = atr,
                signal_strength = abs(score),
                regime          = regime,
            )
            stop = self.risk_manager.calculate_stop_loss(price, atr, 'buy')

            self.paper_trader.execute_buy(
                symbol       = symbol,
                size_usdt    = size,
                price        = price,
                stop_loss    = stop,
                regime       = regime,
                signal_score = score,
                reason       = f'signal:{regime}',
            )

        # ── SELL ─────────────────────────────────────────────────────────
        elif direction == 'sell' and symbol in self.paper_trader.positions:
            self.paper_trader.execute_sell(
                symbol = symbol,
                price  = price,
                reason = f'signal:{regime}',
            )

        else:
            if direction == 'buy' and symbol in self.paper_trader.positions:
                logger.debug("  %s → BUY signal but already in position", symbol)
            elif direction == 'sell' and symbol not in self.paper_trader.positions:
                logger.debug("  %s → SELL signal but no open position", symbol)

    def _print_status(self) -> None:
        """Print a formatted portfolio status summary."""
        status = self.paper_trader.get_status()
        logger.info(
            "\n  💰 Portfolio: $%.2f | Return: %+.2f%% | Drawdown: %.2f%% | "
            "Positions: %d | Trades: %d",
            status['portfolio_value'],
            status['total_return_pct'],
            status['drawdown_pct'],
            status['open_positions'],
            status['total_trades'],
        )

    def run_backtest(self, symbol: str = 'BTC/USDT') -> None:
        """
        Fetch historical data and run a walk-forward backtest.

        Args:
            symbol: Symbol to backtest
        """
        from backtest.engine import BacktestEngine

        logger.info("📊 Fetching historical data for backtest (%s)...", symbol)
        df = self.data_feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=500)
        if df is None or len(df) < 200:
            logger.error("Insufficient data for backtest")
            return

        engine = BacktestEngine(train_periods=100, test_periods=50, step_size=25)
        result = engine.run(df, symbol=symbol, timeframe=PRIMARY_TIMEFRAME)

        print("\n" + "=" * 60)
        print(f"  Backtest Complete: {symbol}")
        print("=" * 60)
        m = result.aggregate_metrics
        if 'error' not in m:
            print(f"  Windows:        {m.get('total_windows', 0)}")
            print(f"  Total Trades:   {m.get('total_trades', 0)}")
            print(f"  Total Return:   {m.get('total_return_pct', 0):+.2f}%")
            print(f"  Sharpe Ratio:   {m.get('sharpe_ratio', 0):.4f}")
            print(f"  Win Rate:       {m.get('win_rate', 0):.1f}%")
            print(f"  Max Drawdown:   {m.get('max_drawdown_pct', 0):.2f}%")
        else:
            print(f"  Error: {m.get('error')}")
        print("=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='MidasTouch Algorithmic Trading Bot')
    parser.add_argument('--dashboard', action='store_true',
                        help='Start the FastAPI dashboard on :8000')
    parser.add_argument('--backtest', action='store_true',
                        help='Run a walk-forward backtest on BTC/USDT and exit')
    parser.add_argument('--symbol', default='BTC/USDT',
                        help='Symbol for backtest (default: BTC/USDT)')
    args = parser.parse_args()

    bot = MidasTouchBot()

    if args.backtest:
        bot.run_backtest(symbol=args.symbol)
        sys.exit(0)

    if args.dashboard:
        inject_dependencies(
            paper_trader    = bot.paper_trader,
            data_feed       = bot.data_feed,
            bot_running_flag = bot.running_flag,
        )
        # Run dashboard in a background thread
        def start_uvicorn():
            uvicorn.run(dashboard_app, host='0.0.0.0', port=8000, log_level='warning')

        dashboard_thread = threading.Thread(target=start_uvicorn, daemon=True)
        dashboard_thread.start()
        logger.info("📡 Dashboard running at http://localhost:8000")

    bot.run_loop()


if __name__ == '__main__':
    main()
