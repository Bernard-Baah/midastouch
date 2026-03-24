# MidasTouch 💰

**An elite algorithmic crypto trading bot — Phase 1**

Multi-strategy, risk-managed, adaptive paper trading system. Built for BTC, ETH, SOL on Binance.

## Architecture

```
midastouch/
├── core/
│   ├── data_feed.py       # Binance OHLCV data (no API key needed)
│   ├── indicators.py      # RSI, MACD, EMA, Bollinger Bands, ATR
│   ├── regime_detector.py # Bull/Bear/Sideways/High-Vol detection
│   ├── signals.py         # Ensemble signal combiner
│   ├── risk_manager.py    # Dynamic position sizing + kill-switch
│   ├── paper_trader.py    # $1,000 virtual trading engine
│   └── performance.py     # Sharpe, Sortino, win rate tracker
├── strategies/
│   ├── trend_following.py # EMA crossover + momentum
│   ├── mean_reversion.py  # RSI + Bollinger Bands
│   └── breakout.py        # Volume breakout detection
├── backtest/
│   └── engine.py          # Walk-forward backtesting
├── dashboard/
│   └── api.py             # FastAPI REST dashboard
├── config.py              # All configuration
└── main.py                # Main bot loop (runs every 5 min)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot (paper trading)
python main.py

# Run dashboard API (separate terminal)
uvicorn dashboard.api:app --host 0.0.0.0 --port 8000

# Run backtest
python backtest/engine.py
```

## Dashboard Endpoints

| Endpoint | Description |
|----------|-------------|
| GET /status | Bot status, capital, drawdown |
| GET /positions | Open positions |
| GET /trades | Trade history |
| GET /performance | Sharpe, Sortino, win rate |
| GET /signals | Current signals per symbol |

## Risk Management

- Dynamic position sizing: 0.5%–7% of capital per trade
- ATR-based stop losses on every trade
- 15% max drawdown kill-switch
- 20% capital always in reserve
- Regime-aware risk adjustment

## Roadmap

- [x] Phase 1: Core engine, paper trading, backtesting
- [ ] Phase 2: ML regime detection, reinforcement learning
- [ ] Phase 3: Live trading (Binance account required)
- [ ] Phase 4: Stock market integration (Alpaca)

## Built By

[Bernard Baah](https://www.bernardbaah.com)
