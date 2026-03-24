"""
MidasTouch - Configuration Constants
All trading parameters, thresholds, and settings.
"""

# ─── Trading Config ────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1000.0
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['15m', '1h', '4h']
PRIMARY_TIMEFRAME = '1h'

# Binance exchange settings
EXCHANGE_ID = 'kucoin'
EXCHANGE_RATE_LIMIT = True

# ─── Risk Management ───────────────────────────────────────────────────────────
MIN_RISK_PER_TRADE = 0.005   # 0.5% minimum risk per trade
MAX_RISK_PER_TRADE = 0.07    # 7% maximum risk per trade
MAX_DRAWDOWN_LIMIT = 0.15    # 15% drawdown kill-switch
RESERVE_CAPITAL_PCT = 0.20   # Keep 20% in reserve always
MAX_CAPITAL_PER_TRADE = 0.80 # Never exceed 80% of available capital
STOP_LOSS_ATR_MULT = 1.5     # Stop-loss = 1.5x ATR from entry

# ─── Signal Thresholds ─────────────────────────────────────────────────────────
MIN_SIGNAL_STRENGTH = 0.6    # Minimum ensemble score to trade

# ─── Indicator Periods ─────────────────────────────────────────────────────────
EMA_PERIODS = [9, 21, 50, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 20
ATR_AVG_PERIOD = 30           # For regime detection

# ─── Breakout Settings ─────────────────────────────────────────────────────────
BREAKOUT_LOOKBACK = 20        # 20-period high/low for breakout
BREAKOUT_VOLUME_MULT = 1.5    # Volume must be 1.5x average

# ─── Data Feed ─────────────────────────────────────────────────────────────────
OHLCV_LIMIT = 300             # Candles to fetch per timeframe
CACHE_TTL_SECONDS = {
    '15m': 60,    # 1-minute cache for 15m bars
    '1h': 240,    # 4-minute cache for 1h bars
    '4h': 900,    # 15-minute cache for 4h bars
}

# ─── Main Loop ─────────────────────────────────────────────────────────────────
LOOP_INTERVAL_SECONDS = 300   # Run every 5 minutes

# ─── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = 'sqlite:///data/trades.db'

# ─── Regime Weights ────────────────────────────────────────────────────────────
# How much each strategy contributes per regime
REGIME_WEIGHTS = {
    'bull': {
        'trend_following': 0.60,
        'mean_reversion':  0.20,
        'breakout':        0.20,
    },
    'bear': {
        'trend_following': 0.20,
        'mean_reversion':  0.50,
        'breakout':        0.30,
    },
    'sideways': {
        'trend_following': 0.20,
        'mean_reversion':  0.60,
        'breakout':        0.20,
    },
    'high_volatility': {
        'trend_following': 0.30,
        'mean_reversion':  0.20,
        'breakout':        0.50,
    },
}

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
