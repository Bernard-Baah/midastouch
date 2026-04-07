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
    SYMBOLS, CRYPTO_SYMBOLS, PRIMARY_TIMEFRAME, INITIAL_CAPITAL,
    QUANT_MODE, QUANT_TOP_N, QUANT_BOTTOM_N,
    MIN_HOLD_BARS, TRAILING_STOP_ENABLED,
)
from core.short_tracker import ShortTracker

from core.data_feed      import DataFeed
from core.stock_feed     import StockFeed
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
    from quant.risk      import size_position, check_drawdown, get_volatility, calculate_trailing_stop

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

def _run_quant_loop(feed: DataFeed, trader: PaperTrader, short_tracker, stock_feed=None) -> None:
    """
    Multi-factor, portfolio-ranked long/short execution.
    Pipeline: DATA → FEATURES → ALPHA → ENSEMBLE → PORTFOLIO → RISK → EXECUTE
    """
    from config import CRYPTO_SYMBOLS
    symbols = CRYPTO_SYMBOLS

    scores:        dict = {}
    price_history: dict = {}
    dfs:           dict = {}
    vols:          dict = {}

    # ── 1. Fetch & feature-engineer all symbols ──────────────────────────────
    for symbol in symbols:
        try:
            # Primary timeframe for signals
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
            vols[symbol] = get_volatility(df)

        except Exception as exc:
            logger.exception("Data/feature error for %s: %s", symbol, exc)

    # ── 1b. Fetch stock data via Alpaca ──────────────────────────────────────
    if stock_feed is not None:
        from config import STOCK_SYMBOLS
        from core.indicators import calculate_indicators as calc_ind
        for symbol in STOCK_SYMBOLS:
            try:
                df = stock_feed.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=200)
                if df is None or len(df) < 20:
                    logger.debug("%s: no stock data (market may be closed)", symbol)
                    continue
                df = calc_ind(df)
                if df is None or df.empty:
                    continue
                df.dropna(inplace=True)
                if df.empty:
                    continue
                df = add_all_features(df)
                dfs[symbol] = df
                price_history[symbol] = df["close"]
                vols[symbol] = get_volatility(df)
                logger.info("Stock %s loaded: %d bars, latest=$%.2f", symbol, len(df), df["close"].iloc[-1])
            except Exception as exc:
                logger.warning("Stock data error %s: %s", symbol, exc)

    if not dfs:
        logger.warning("No valid data — skipping loop")
        return

    # ── 2. Alpha extraction & ensemble scoring ───────────────────────────────
    for symbol, df in dfs.items():
        try:
            signals = extract_alpha_signals(df)
            score   = compute_ensemble_score(signals)
            scores[symbol] = score
            logger.info(
                "%-14s $%-10.2f score=%+.3f [mom=%+.2f rev=%+.2f vol=%.2f vspike=%.2f]",
                symbol, df["close"].iloc[-1], score,
                signals["momentum"], signals["mean_rev"],
                signals["volatility"], signals["volume"],
            )
        except Exception as exc:
            logger.exception("Alpha error for %s: %s", symbol, exc)

    # ── 3. Portfolio construction ─────────────────────────────────────────────
    longs, shorts = select_portfolio(
        scores,
        price_history=price_history,
        top_n=QUANT_TOP_N,
        bottom_n=QUANT_BOTTOM_N,
        separate_universes=True,
    )
    logger.info("Portfolio → LONG: %s | SHORT: %s", longs, shorts)

    # ── 4. Close longs no longer in target portfolio (with min hold time) ────
    for sym in list(trader.positions.keys()):
        if sym not in set(longs):
            pos = trader.positions[sym]
            # Check if held long enough
            open_time_str = pos.get("open_time", "")
            hold_bars = 0
            if open_time_str:
                try:
                    open_dt = datetime.fromisoformat(open_time_str.replace("Z", ""))
                    if open_dt.tzinfo is None:
                        open_dt = open_dt.replace(tzinfo=timezone.utc)
                    elapsed_hours = (datetime.now(timezone.utc) - open_dt).total_seconds() / 3600
                    hold_bars = int(elapsed_hours)  # 1h bars
                except Exception:
                    hold_bars = MIN_HOLD_BARS  # default: allow close

            if hold_bars >= MIN_HOLD_BARS:
                price = dfs[sym]["close"].iloc[-1] if sym in dfs else None
                if price:
                    trader.execute_sell(sym, price, reason="portfolio_rebalance")
            else:
                logger.debug("%s held for %d bars — min hold not reached, keeping", sym, hold_bars)

    # ── 5. Close shorts no longer in target ──────────────────────────────────
    for sym in list(short_tracker.shorts.keys()):
        if sym not in set(shorts):
            price = dfs[sym]["close"].iloc[-1] if sym in dfs else None
            if price:
                short_tracker.close_short(sym, price, reason="portfolio_rebalance")

    # ── 6. Open new long positions ────────────────────────────────────────────
    for symbol in longs:
        if symbol in trader.positions or symbol not in dfs:
            continue
        df    = dfs[symbol]
        vol   = vols.get(symbol, 0.02)
        size  = size_position(trader.cash, vol)
        price = df["close"].iloc[-1]

        if size >= 10 and size <= trader.cash * 0.8:
            stop = price * (1 - vol * 3.5)  # wider stop
            trader.execute_buy(
                symbol=symbol, size_usdt=size, price=price,
                stop_loss=stop, regime=detect_regime(df),
                signal_score=scores.get(symbol, 0.0),
                reason=f"quant_long_{scores.get(symbol, 0):.3f}",
            )

    # ── 7. Open new short positions ───────────────────────────────────────────
    for symbol in shorts:
        if symbol in short_tracker.shorts or symbol not in dfs:
            continue
        df    = dfs[symbol]
        vol   = vols.get(symbol, 0.02)
        size  = size_position(trader.cash, vol)
        price = df["close"].iloc[-1]

        if size >= 10:
            stop = price * (1 + vol * 2.0)   # short stop = price RISES
            short_tracker.open_short(
                symbol=symbol, entry_price=price,
                size_usdt=size, stop_loss=stop,
                reason=f"quant_short_{scores.get(symbol, 0):.3f}",
            )

    # ── 8. Update trailing stops + check stop losses ──────────────────────────
    for symbol in list(trader.positions.keys()):
        if symbol not in dfs:
            continue
        current_price = dfs[symbol]["close"].iloc[-1]
        pos = trader.positions[symbol]

        # Update trailing stop if enabled
        if TRAILING_STOP_ENABLED:
            peak = max(pos.get("peak_price", pos["entry_price"]), current_price)
            pos["peak_price"] = peak
            new_stop = calculate_trailing_stop(pos["entry_price"], peak, "long")
            if new_stop > pos["stop_loss"]:
                pos["stop_loss"] = new_stop
                logger.debug("%s trailing stop updated to %.4f", symbol, new_stop)

        trader.check_stop_losses(symbol, current_price)

    for symbol in list(short_tracker.shorts.keys()):
        if symbol in dfs:
            short_tracker.check_stop_losses(symbol, dfs[symbol]["close"].iloc[-1])


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

    feed          = DataFeed()
    trader        = PaperTrader()
    risk          = RiskManager()
    short_tracker = ShortTracker()
    stock_feed    = StockFeed()
    loop          = 0

    logger.info("Starting capital: $%.2f | symbols: %s | mode: %s",
                INITIAL_CAPITAL, CRYPTO_SYMBOLS if QUANT_MODE else SYMBOLS, "QUANT" if QUANT_MODE else "LEGACY")

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
                _run_quant_loop(feed, trader, short_tracker, stock_feed=stock_feed)
            else:
                _run_legacy_loop(feed, trader, risk)
        except Exception as exc:
            logger.exception("Unhandled loop error: %s", exc)

        # ── Status summary ───────────────────────────────────────────────────
        status    = trader.get_status()
        capital   = status.get("portfolio_value", INITIAL_CAPITAL)
        ret_pct   = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        dd_pct    = status.get("drawdown_pct", 0) * 100
        n_pos     = len(trader.positions)
        n_shorts  = len(short_tracker.shorts)
        short_pnl = short_tracker.get_unrealized_pnl(
            {sym: 0 for sym in short_tracker.shorts}  # prices checked per-loop above
        )

        logger.info(
            "📊 Capital=$%.2f  Return=%+.2f%%  Drawdown=%.1f%%  Longs=%d  Shorts=%d  ShortPnL=$%.2f",
            capital, ret_pct, dd_pct, n_pos, n_shorts, short_pnl,
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
