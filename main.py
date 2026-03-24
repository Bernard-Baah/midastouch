"""
MidasTouch — Main Bot Loop
Runs every 5 minutes. Fetches data, generates signals, executes paper trades.
"""

import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import SYMBOLS, PRIMARY_TIMEFRAME, MAX_DRAWDOWN_LIMIT, INITIAL_CAPITAL
from core.data_feed import DataFeed
from core.indicators import add_indicators
from core.regime_detector import detect_regime
from core.signals import get_ensemble_signal
from core.risk_manager import calculate_position_size, calculate_stop_loss, check_drawdown
from core.paper_trader import PaperTrader
from core.performance import calculate_all

LOOP_INTERVAL = 300  # 5 minutes


def print_header():
    print("\n" + "="*55)
    print("  💰 MidasTouch — Paper Trading Bot (Phase 1)")
    print("="*55)


def run():
    print_header()
    feed = DataFeed()
    trader = PaperTrader(initial_capital=INITIAL_CAPITAL)
    kill_switch = False
    loop = 0

    print(f"  Starting capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Timeframe: {PRIMARY_TIMEFRAME}")
    print(f"  Loop interval: {LOOP_INTERVAL}s")
    print("="*55)

    while not kill_switch:
        loop += 1
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"\n[Loop {loop}] {now}")

        # Check kill switch
        summary = trader.summary()
        if check_drawdown(summary['total_capital'], trader.peak_capital, MAX_DRAWDOWN_LIMIT):
            print(f"\n🚨 KILL SWITCH TRIGGERED — Drawdown exceeded {MAX_DRAWDOWN_LIMIT*100:.0f}%")
            print(f"   Capital: ${summary['total_capital']:.2f} (started ${INITIAL_CAPITAL:.2f})")
            kill_switch = True
            break

        current_prices = {}
        signals_snapshot = {}

        for symbol in SYMBOLS:
            try:
                # Fetch data
                df = feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=200)
                if df is None or len(df) < 50:
                    print(f"  ⚠️  {symbol}: insufficient data")
                    continue

                df = add_indicators(df)
                df.dropna(inplace=True)

                if df.empty:
                    continue

                current_price = df['close'].iloc[-1]
                current_prices[symbol] = current_price

                # Detect regime
                regime = detect_regime(df)

                # Generate signal
                signal = get_ensemble_signal(df)
                signals_snapshot[symbol] = {
                    'price': round(current_price, 4),
                    'regime': regime,
                    'direction': signal['direction'],
                    'score': round(signal['score'], 3),
                }

                print(f"  {symbol}: ${current_price:.2f} | Regime: {regime} | Signal: {signal['direction']} ({signal['score']:.2f})")

                # Execute trades
                if signal['direction'] == 'buy' and symbol not in trader.positions:
                    atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                    size = calculate_position_size(
                        trader.cash, atr, signal['score'], regime
                    )
                    if size > 10:
                        stop = calculate_stop_loss(current_price, atr, 'buy')
                        trader.execute_buy(symbol, size, current_price, stop, reason=f"ensemble_{signal['score']:.2f}")

                elif signal['direction'] == 'sell' and symbol in trader.positions:
                    trader.execute_sell(symbol, current_price, reason=f"ensemble_{signal['score']:.2f}")

            except Exception as e:
                print(f"  ❌ Error processing {symbol}: {e}")

        # Check stop losses
        if current_prices:
            trader.check_stop_losses(current_prices)

        # Print portfolio summary
        summary = trader.summary()
        print(f"\n  📊 Portfolio: ${summary['total_capital']:.2f} | Return: {summary['total_return_pct']:+.2f}% | Drawdown: {summary['drawdown_pct']:.1f}%")
        print(f"  Positions: {summary['open_positions']} open | Trades: {summary['total_trades']} total")

        if loop % 12 == 0:  # Every hour, print performance
            perf = calculate_all()
            print(f"\n  📈 Performance | Sharpe: {perf['sharpe']} | Win Rate: {perf['win_rate']}% | PnL: ${perf.get('total_pnl', 0):.2f}")

        print(f"  Next loop in {LOOP_INTERVAL}s...")
        time.sleep(LOOP_INTERVAL)

    print("\n✅ MidasTouch stopped.")


if __name__ == '__main__':
    run()
