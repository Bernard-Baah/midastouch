"""
MidasTouch — Main Bot Loop  (v2 — Quant Mode)
Two modes:
  QUANT_MODE = True  → multi-factor, portfolio-ranked long/short
  QUANT_MODE = False → legacy single-asset signal flow
"""

import time
import logging
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SYMBOLS, PRIMARY_TIMEFRAME, INITIAL_CAPITAL,
    QUANT_MODE, QUANT_TOP_N, QUANT_BOTTOM_N,
)

from core.data_feed      import DataFeed
from core.indicators     import calculate_indicators
from core.regime_detector import detect_regime
from core.signals        import generate_ensemble_signal
from core.risk_manager   import RiskManager
from core.paper_trader   import PaperTrader
from core.performance    import calculate_all

# Quant layer imports (only used when QUANT_MODE = True)
if QUANT_MODE:
    from quant.features  import add_all_features
    from quant.alpha     import extract_alpha_signals
    from quant.ensemble  import compute_ensemble_score
    from quant.portfolio import select_portfolio
    from quant.risk      import size_position, check_drawdown, get_volatility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("midastouch.main")

LOOP_INTERVAL = 300   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_header():
    mode = "QUANT MODE" if QUANT_MODE else "LEGACY MODE"
    print("\n" + "=" * 58)
    print(f"  💰 MidasTouch — Paper Trading Bot  [{mode}]")
    print("=" * 58)


# ─────────────────────────────────────────────────────────────────────────────
# Quant loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_quant_loop(feed: DataFeed, trader: PaperTrader) -> None:
    """
    Multi-factor, portfolio-ranked long/short execution.

    Pipeline per loop:
      DATA → FEATURES → ALPHA → ENSEMBLE → PORTFOLIO → RISK → EXECUTE
    """
    scores:        dict[str, float] = {}
    price_history: dict            = {}
    dfs:           dict            = {}

    # ── 1. Fetch & feature-engineer all symbols ─────────────────────────────
    for symbol in SYMBOLS:
        try:
            df = feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=200)
            if df is None or len(df) < 50:
                logger.warning("%s: insufficient data — skipping", symbol)
                continue

            df = calculate_indicators(df)
            if df is None or df.empty:
                continue
            df.dropna(inplace=True)
            if df.empty:
                continue

            df = add_all_features(df)
            dfs[symbol] = df
            price_history[symbol] = df["close"]

        except Exception as exc:
            logger.exception("Data/feature error for %s: %s", symbol, exc)

    if not dfs:
        logger.warning("No valid data for any symbol — skipping loop")
        return

    # ── 2. Alpha extraction & ensemble scoring ───────────────────────────────
    for symbol, df in dfs.items():
        try:
            signals = extract_alpha_signals(df)
            score   = compute_ensemble_score(signals)
            scores[symbol] = score
            logger.info(
                "%-12s price=$%.2f score=%+.3f  [mom=%+.2f rev=%+.2f vol=%.2f vols=%.2f]",
                symbol,
                df["close"].iloc[-1],
                score,
                signals["momentum"],
                signals["mean_rev"],
                signals["volatility"],
                signals["volume"],
            )
        except Exception as exc:
            logger.exception("Alpha/ensemble error for %s: %s", symbol, exc)

    # ── 3. Portfolio construction ─────────────────────────────────────────────
    longs, shorts = select_portfolio(
        scores,
        price_history=price_history,
        top_n=QUANT_TOP_N,
        bottom_n=QUANT_BOTTOM_N,
    )
    logger.info("Portfolio → LONG: %s | SHORT: %s", longs, shorts)

    # ── 4. Close positions that are no longer in the portfolio ────────────────
    open_symbols = set(trader.positions.keys())
    target_longs = set(longs)

    for sym in list(open_symbols):
        if sym not in target_longs:
            price = dfs[sym]["close"].iloc[-1] if sym in dfs else None
            if price:
                trader.execute_sell(sym, price, reason="portfolio_rebalance")

    # ── 5. Open new long positions ────────────────────────────────────────────
    for symbol in longs:
        if symbol in trader.positions:
            continue   # already held
        if symbol not in dfs:
            continue

        df   = dfs[symbol]
        vol  = get_volatility(df)
        size = size_position(trader.cash, vol)
        price = df["close"].iloc[-1]

        if size >= 10 and size <= trader.cash:
            stop = price * (1 - vol * 1.5)   # ATR-proxy stop
            trader.execute_buy(
                symbol=symbol,
                size_usdt=size,
                price=price,
                stop_loss=stop,
                regime=detect_regime(df),
                signal_score=scores.get(symbol, 0.0),
                reason=f"quant_long_{scores.get(symbol, 0.0):.3f}",
            )

    # ── 6. Short positions (paper: simulated as sell-then-track) ─────────────
    # Note: true short execution requires margin. In paper mode we log intent.
    for symbol in shorts:
        logger.info("SHORT candidate: %s (score=%.3f) — logged only in paper mode", symbol, scores.get(symbol, 0.0))

    # ── 7. Check stop losses ──────────────────────────────────────────────────
    for symbol in list(trader.positions.keys()):
        if symbol in dfs:
            trader.check_stop_losses(symbol, dfs[symbol]["close"].iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Legacy loop (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _run_legacy_loop(feed: DataFeed, trader: PaperTrader, risk: RiskManager) -> None:
    for symbol in SYMBOLS:
        try:
            df = feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=200)
            if df is None or len(df) < 50:
                continue
            df = calculate_indicators(df)
            if df is None or df.empty:
                continue
            df.dropna(inplace=True)
            if df.empty:
                continue

            current_price = df["close"].iloc[-1]
            regime        = detect_regime(df)
            signal        = generate_ensemble_signal(df, regime)

            print(f"  {symbol}: ${current_price:.2f} | {regime} | {signal['direction']} ({signal['score']:.2f})")

            trader.check_stop_losses(symbol, current_price)

            if signal["direction"] == "buy" and symbol not in trader.positions:
                atr  = df["atr_14"].iloc[-1] if "atr_14" in df.columns else current_price * 0.02
                size = risk.calculate_position_size(trader.cash, atr, signal["score"], regime)
                if size > 10:
                    stop = risk.calculate_stop_loss(current_price, atr, "buy")
                    trader.execute_buy(symbol=symbol, size_usdt=size, price=current_price,
                                       stop_loss=stop, regime=regime,
                                       signal_score=signal["score"],
                                       reason=f"legacy_{signal['score']:.2f}")
            elif signal["direction"] == "sell" and symbol in trader.positions:
                trader.execute_sell(symbol, current_price,
                                    reason=f"legacy_{signal['score']:.2f}")

        except Exception as exc:
            logger.exception("Legacy loop error %s: %s", symbol, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    _print_header()

    feed   = DataFeed()
    trader = PaperTrader()
    risk   = RiskManager()
    loop   = 0

    logger.info("Starting capital: $%.2f | symbols: %s | mode: %s",
                INITIAL_CAPITAL, SYMBOLS, "QUANT" if QUANT_MODE else "LEGACY")

    while True:
        loop += 1
        logger.info("──── Loop %d  %s ────", loop, _now_utc())

        # ── Kill-switch ──────────────────────────────────────────────────────
        status  = trader.get_status()
        capital = status.get("portfolio_value", status.get("cash", INITIAL_CAPITAL))

        if QUANT_MODE:
            triggered = check_drawdown(capital, trader.peak_capital)
        else:
            triggered = risk.check_drawdown(capital, trader.peak_capital)

        if triggered:
            logger.critical("KILL SWITCH — halting bot. Capital=$%.2f", capital)
            break

        # ── Execute loop ─────────────────────────────────────────────────────
        try:
            if QUANT_MODE:
                _run_quant_loop(feed, trader)
            else:
                _run_legacy_loop(feed, trader, risk)
        except Exception as exc:
            logger.exception("Unhandled loop error: %s", exc)

        # ── Status summary ───────────────────────────────────────────────────
        status   = trader.get_status()
        capital  = status.get("portfolio_value", INITIAL_CAPITAL)
        ret_pct  = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        dd_pct   = status.get("drawdown_pct", 0) * 100
        n_pos    = len(trader.positions)

        logger.info(
            "📊 Capital=$%.2f  Return=%+.2f%%  Drawdown=%.1f%%  Positions=%d",
            capital, ret_pct, dd_pct, n_pos,
        )

        if loop % 12 == 0:
            perf = calculate_all()
            logger.info(
                "📈 WinRate=%.1f%%  Sharpe=%.2f  PnL=$%.2f  Trades=%d",
                perf.get("win_rate", 0),
                perf.get("sharpe", 0),
                perf.get("total_pnl", 0),
                perf.get("total_trades", 0),
            )

        logger.info("⏳ Next loop in %ds", LOOP_INTERVAL)
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()
