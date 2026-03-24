# 🟡 MidasTouch — Algorithmic Crypto Trading Bot

A professional-grade Python algorithmic trading bot with ensemble signal generation,
ATR-based risk management, and a paper trading engine.

---

## Features

| Component | Description |
|-----------|-------------|
| **Multi-timeframe data** | 15m / 1h / 4h from Binance (no API key needed) |
| **Technical indicators** | EMA (9/21/50/200), RSI, MACD, Bollinger Bands, ATR |
| **Market regime detection** | bull / bear / sideways / high_volatility |
| **3 strategies** | Trend Following, Mean Reversion, Breakout |
| **Ensemble signals** | Regime-weighted combination of all 3 strategies |
| **ATR-based risk sizing** | Dynamic position sizing proportional to volatility |
| **Paper trading engine** | $1,000 virtual capital with full P&L tracking |
| **SQLite trade log** | Every trade persisted with entry/exit/P&L/regime |
| **Walk-forward backtest** | Rolling train/test windows to avoid look-ahead bias |
| **FastAPI dashboard** | REST API for real-time monitoring |
| **Drawdown kill-switch** | Halts trading if drawdown exceeds 15% |

---

## Quick Start

### 1. Install Dependencies

```bash
cd midastouch
pip install -r requirements.txt
```

### 2. Run the Bot (paper trading only)

```bash
python main.py
```

### 3. Run with Dashboard (REST API on :8000)

```bash
python main.py --dashboard
```

Then open: http://localhost:8000

### 4. Run a Backtest

```bash
python main.py --backtest
python main.py --backtest --symbol ETH/USDT
```

---

## Project Structure

```
midastouch/
├── core/
│   ├── data_feed.py        # Binance OHLCV fetcher with caching
│   ├── indicators.py       # EMA, RSI, MACD, BB, ATR, Volume SMA
│   ├── regime_detector.py  # Market regime classification
│   ├── signals.py          # Ensemble signal combiner
│   ├── risk_manager.py     # ATR position sizing + stop-loss + kill-switch
│   ├── paper_trader.py     # Paper trading engine (SQLite backed)
│   └── performance.py      # Sharpe, Sortino, win rate, drawdown
├── strategies/
│   ├── trend_following.py  # EMA crossover + MACD + RSI momentum
│   ├── mean_reversion.py   # RSI + Bollinger Band reversion
│   └── breakout.py         # Volume-confirmed price breakout
├── backtest/
│   └── engine.py           # Walk-forward backtesting framework
├── dashboard/
│   └── api.py              # FastAPI REST dashboard
├── data/
│   └── trades.db           # SQLite trade log (auto-created)
├── config.py               # All configuration constants
├── main.py                 # Main bot orchestrator
└── requirements.txt
```

---

## Dashboard Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /status` | Bot status, capital, drawdown |
| `GET /positions` | Open positions |
| `GET /trades?limit=50` | Trade history |
| `GET /performance` | Sharpe, Sortino, win rate, P&L |
| `GET /signals` | Current signals per symbol |
| `GET /health` | Health check |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL` | $1,000 | Starting virtual capital |
| `SYMBOLS` | BTC/USDT, ETH/USDT, SOL/USDT | Traded pairs |
| `PRIMARY_TIMEFRAME` | 1h | Main signal timeframe |
| `MIN_RISK_PER_TRADE` | 0.5% | Minimum risk per trade |
| `MAX_RISK_PER_TRADE` | 7% | Maximum risk per trade |
| `MAX_DRAWDOWN_LIMIT` | 15% | Kill-switch threshold |
| `MIN_SIGNAL_STRENGTH` | 0.6 | Minimum ensemble score to trade |
| `STOP_LOSS_ATR_MULT` | 1.5x | Stop-loss distance in ATR units |

---

## Strategy Signal Logic

### Trend Following
- **BUY**: EMA9 > EMA21 > EMA50, price > EMA200, RSI 50–65, MACD line > signal
- **SELL**: EMA9 < EMA21 or RSI > 70 or bearish MACD crossover

### Mean Reversion
- **BUY**: RSI < 35, price below lower Bollinger Band, high volume
- **SELL**: RSI > 65, price above upper Bollinger Band

### Breakout
- **BUY**: Price breaks above 20-period high with 1.5x average volume
- **SELL**: Price breaks below 20-period low

### Ensemble
Strategies are weighted by market regime:
- **Bull market**: Trend Following 60%, Breakout 20%, Mean Reversion 20%
- **Bear market**: Mean Reversion 50%, Breakout 30%, Trend Following 20%
- **Sideways**: Mean Reversion 60%, Trend Following 20%, Breakout 20%
- **High volatility**: Breakout 50%, Trend Following 30%, Mean Reversion 20%

---

## Risk Management

- **ATR-based position sizing**: Risk amount ÷ ATR = units to hold
- **Signal-scaled risk**: Stronger signals → larger positions (within bounds)
- **Regime-adjusted risk**: Bear/volatile regimes → reduced exposure
- **Reserve capital**: 20% always kept in cash
- **Drawdown kill-switch**: All positions closed if portfolio drops 15% from peak

---

## Notes

- This is **paper trading only** — no real money is ever at risk
- Binance public endpoints are used (no API key required)
- All trades are logged to `data/trades.db` (SQLite)
- The bot runs every 5 minutes by default

---

*Built with ❤️ by the MidasTouch project*
