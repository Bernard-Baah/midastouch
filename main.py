"""
MidasTouch v3 — Strict Quant Bot
Rules: Max 3 positions | Min score 0.15 | Pause after 3 losses | Force close 48h
"""

import time
import logging
import sys
import os
import sqlite3
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from config import (INITIAL_CAPITAL, QUANT_TOP_N, QUANT_BOTTOM_N, PRIMARY_TIMEFRAME)
from core.data_feed       import DataFeed
from core.indicators      import calculate_indicators
from core.regime_detector import detect_regime
from core.paper_trader    import PaperTrader
from core.performance     import calculate_all
from core.stock_feed      import StockFeed
from core.short_tracker   import ShortTracker
from quant.features       import add_all_features
from quant.alpha          import extract_alpha_signals
from quant.ensemble       import compute_ensemble_score
from quant.portfolio      import select_portfolio
from quant.risk           import size_position, check_drawdown, get_volatility

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("midastouch.main")

# ── STRICT RULES v3 ───────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS     = 3
MIN_SCORE_TO_TRADE     = 0.15
MAX_CONSECUTIVE_LOSSES = 3
MAX_HOLD_HOURS         = 48
MIN_HOLD_HOURS         = 6
LOOP_INTERVAL          = 300

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'trades.db')


def get_consecutive_losses() -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT pnl FROM trades WHERE status='closed' ORDER BY id DESC LIMIT 10")
        rows = cur.fetchall()
        conn.close()
        count = 0
        for row in rows:
            if row[0] is not None and row[0] < 0:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def get_open_hours(pos: dict) -> float:
    try:
        s = pos.get("open_time", "")
        if not s:
            return 0
        dt = datetime.fromisoformat(s.replace("Z", ""))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return 0


def force_close_old(trader, short_tracker, dfs):
    for sym in list(trader.positions.keys()):
        h = get_open_hours(trader.positions[sym])
        if h >= MAX_HOLD_HOURS and sym in dfs:
            price = dfs[sym]["close"].iloc[-1]
            logger.info("⏰ Force closing %s after %.0fh", sym, h)
            trader.execute_sell(sym, price, reason=f"max_hold_{h:.0f}h")
    for sym in list(short_tracker.shorts.keys()):
        h = get_open_hours(short_tracker.shorts[sym])
        if h >= MAX_HOLD_HOURS and sym in dfs:
            price = dfs[sym]["close"].iloc[-1]
            short_tracker.close_short(sym, price, reason=f"max_hold_{h:.0f}h")


def run():
    print("\n" + "="*60)
    print("  💰 MidasTouch v3 — Strict Rules (fresh start)")
    print(f"  Max positions: {MAX_OPEN_POSITIONS} | Min score: {MIN_SCORE_TO_TRADE} | Max hold: {MAX_HOLD_HOURS}h")
    print("="*60)

    feed          = DataFeed()
    stock_feed    = StockFeed()
    trader        = PaperTrader()
    short_tracker = ShortTracker()
    loop          = 0

    while True:
        loop += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info("──── Loop %d  %s ────", loop, now)

        status  = trader.get_status()
        capital = status.get("portfolio_value", INITIAL_CAPITAL)
        if check_drawdown(capital, trader.peak_capital):
            logger.critical("🚨 KILL SWITCH — halting")
            break

        dfs = {}; scores = {}; price_history = {}; vols = {}

        from config import CRYPTO_SYMBOLS, STOCK_SYMBOLS

        for sym in CRYPTO_SYMBOLS:
            try:
                df = feed.fetch_ohlcv(sym, PRIMARY_TIMEFRAME, limit=200)
                if df is None or len(df) < 50: continue
                df = calculate_indicators(df)
                if df is None or df.empty: continue
                df.dropna(inplace=True)
                df = add_all_features(df)
                dfs[sym] = df; price_history[sym] = df["close"]; vols[sym] = get_volatility(df)
            except Exception as e:
                logger.warning("%s: %s", sym, e)

        for sym in STOCK_SYMBOLS:
            try:
                df = stock_feed.fetch_ohlcv(sym, PRIMARY_TIMEFRAME, limit=200)
                if df is None or len(df) < 20: continue
                df = calculate_indicators(df)
                if df is None or df.empty: continue
                df.dropna(inplace=True)
                df = add_all_features(df)
                dfs[sym] = df; price_history[sym] = df["close"]; vols[sym] = get_volatility(df)
            except Exception as e:
                logger.debug("%s: %s", sym, e)

        if not dfs:
            logger.warning("No data — skipping")
            time.sleep(LOOP_INTERVAL); continue

        for sym, df in dfs.items():
            try:
                scores[sym] = compute_ensemble_score(extract_alpha_signals(df))
                logger.info("%-14s $%-10.2f score=%+.3f", sym, df["close"].iloc[-1], scores[sym])
            except Exception as e:
                logger.warning("Score %s: %s", sym, e)

        force_close_old(trader, short_tracker, dfs)

        for sym in list(trader.positions.keys()):
            if sym in dfs: trader.check_stop_losses(sym, dfs[sym]["close"].iloc[-1])
        for sym in list(short_tracker.shorts.keys()):
            if sym in dfs: short_tracker.check_stop_losses(sym, dfs[sym]["close"].iloc[-1])

        longs, shorts = select_portfolio(scores, price_history=price_history,
                                         top_n=QUANT_TOP_N, bottom_n=QUANT_BOTTOM_N,
                                         separate_universes=True)

        total_open = len(trader.positions) + len(short_tracker.shorts)
        consec     = get_consecutive_losses()
        paused     = consec >= MAX_CONSECUTIVE_LOSSES

        if paused:
            logger.warning("🚫 PAUSED — %d consecutive losses", consec)

        for sym in list(trader.positions.keys()):
            if sym not in set(longs) and get_open_hours(trader.positions[sym]) >= MIN_HOLD_HOURS:
                if sym in dfs: trader.execute_sell(sym, dfs[sym]["close"].iloc[-1], reason="rebalance")

        for sym in list(short_tracker.shorts.keys()):
            if sym not in set(shorts) and get_open_hours(short_tracker.shorts[sym]) >= MIN_HOLD_HOURS:
                if sym in dfs: short_tracker.close_short(sym, dfs[sym]["close"].iloc[-1], reason="rebalance")

        if not paused:
            for sym in longs:
                total_open = len(trader.positions) + len(short_tracker.shorts)
                if total_open >= MAX_OPEN_POSITIONS: break
                if sym in trader.positions or sym not in dfs: continue
                score = scores.get(sym, 0)
                if abs(score) < MIN_SCORE_TO_TRADE:
                    logger.info("Score too low %s (%.3f) — skip", sym, score); continue
                vol = vols.get(sym, 0.02)
                size = size_position(trader.cash, vol)
                price = dfs[sym]["close"].iloc[-1]
                if size >= 10 and size <= trader.cash * 0.4:
                    stop = price * (1 - vol * 4.0)
                    trader.execute_buy(sym, size, price, stop, regime=detect_regime(dfs[sym]),
                                       signal_score=score, reason=f"long_{score:.3f}")

            for sym in shorts:
                total_open = len(trader.positions) + len(short_tracker.shorts)
                if total_open >= MAX_OPEN_POSITIONS: break
                if sym in short_tracker.shorts or sym not in dfs: continue
                score = scores.get(sym, 0)
                if abs(score) < MIN_SCORE_TO_TRADE: continue
                vol = vols.get(sym, 0.02)
                size = size_position(trader.cash, vol)
                price = dfs[sym]["close"].iloc[-1]
                if size >= 10:
                    stop = price * (1 + vol * 4.0)
                    short_tracker.open_short(sym, price, size, stop, reason=f"short_{score:.3f}")

        status  = trader.get_status()
        capital = status.get("portfolio_value", INITIAL_CAPITAL)
        ret_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        logger.info("📊 Capital=$%.2f  Return=%+.2f%%  Longs=%d  Shorts=%d  Losses=%d/%d",
                    capital, ret_pct, len(trader.positions), len(short_tracker.shorts),
                    consec, MAX_CONSECUTIVE_LOSSES)

        if loop % 12 == 0:
            perf = calculate_all()
            logger.info("📈 WinRate=%.1f%%  PnL=$%.2f  Trades=%d",
                        perf.get("win_rate",0), perf.get("total_pnl",0), perf.get("total_trades",0))

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()
