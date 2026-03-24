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
from core.indicators import calculate_indicators
from core.regime_detector import detect_regime
from core.signals import generate_ensemble_signal
from core.risk_manager import RiskManager
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
    trader = PaperTrader()
    risk = RiskManager()
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
        status = trader.get_status()
        total_capital = status['cash'] + sum(
            p.get('current_value', p['size_usdt']) for p in trader.positions.values()
        )
        if risk.check_drawdown(total_capital, trader.peak_capital):
            print(f"\n🚨 KILL SWITCH — Drawdown exceeded {MAX_DRAWDOWN_LIMIT*100:.0f}%")
            print(f"   Capital: ${total_capital:.2f}")
            kill_switch = True
            break

        for symbol in SYMBOLS:
            try:
                df = feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=200)
                if df is None or len(df) < 50:
                    print(f"  ⚠️  {symbol}: insufficient data")
                    continue

                df = calculate_indicators(df)
                if df is None or df.empty:
                    continue
                df.dropna(inplace=True)
                if df.empty:
                    continue

                current_price = df['close'].iloc[-1]
                regime = detect_regime(df)
                signal = generate_ensemble_signal(df, regime)

                print(f"  {symbol}: ${current_price:.2f} | {regime} | {signal['direction']} ({signal['score']:.2f})")

                # Check stop loss for this symbol
                trader.check_stop_losses(symbol, current_price)

                # Buy signal
                if signal['direction'] == 'buy' and symbol not in trader.positions:
                    atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else current_price * 0.02
                    size = risk.calculate_position_size(trader.cash, atr, signal['score'], regime)
                    if size > 10:
                        stop = risk.calculate_stop_loss(current_price, atr, 'buy')
                        trader.execute_buy(
                            symbol=symbol,
                            size_usdt=size,
                            price=current_price,
                            stop_loss=stop,
                            regime=regime,
                            signal_score=signal['score'],
                            reason=f"ensemble_{signal['score']:.2f}"
                        )

                # Sell signal
                elif signal['direction'] == 'sell' and symbol in trader.positions:
                    trader.execute_sell(symbol, current_price, reason=f"ensemble_{signal['score']:.2f}")

            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
                import traceback; traceback.print_exc()

        # Summary
        status = trader.get_status()
        cash = status['cash']
        total = cash + sum(p['size_usdt'] for p in trader.positions.values())
        ret_pct = (total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        dd_pct = status.get('drawdown_pct', 0) * 100

        print(f"\n  📊 Capital: ${total:.2f} | Return: {ret_pct:+.2f}% | Drawdown: {dd_pct:.1f}%")
        print(f"  Positions: {len(trader.positions)} open | Cash: ${cash:.2f}")

        if loop % 12 == 0:
            perf = calculate_all()
            print(f"\n  📈 Sharpe: {perf['sharpe']} | Win Rate: {perf['win_rate']}% | PnL: ${perf.get('total_pnl', 0):.2f}")

        print(f"  ⏳ Next loop in {LOOP_INTERVAL}s...")
        time.sleep(LOOP_INTERVAL)

    print("\n✅ MidasTouch stopped.")


if __name__ == '__main__':
    run()
