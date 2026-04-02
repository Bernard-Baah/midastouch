# MidasTouch 💰

**An institutional-grade algorithmic trading system — multi-asset, multi-factor, long/short**

Built by [Bernard Baah](https://www.bernardbaah.com)

---

## Live Status

- **Mode:** Quant (multi-factor ensemble)
- **Paper Trading:** $1,000 virtual crypto + $100,000 Alpaca stocks
- **Assets:** 6 crypto + 6 equities
- **Exchange:** KuCoin (crypto) · Alpaca Markets (stocks)
- **Dashboard:** `http://184.72.110.185:8000`

---

## Architecture

```
DATA → FEATURE ENGINEERING → ALPHA SIGNALS → ENSEMBLE MODEL → PORTFOLIO CONSTRUCTION → RISK MANAGEMENT → EXECUTION
```

### Module Structure

```
midastouch/
├── core/
│   ├── data_feed.py        # KuCoin OHLCV data (crypto)
│   ├── stock_feed.py       # Alpaca REST API (equities)
│   ├── indicators.py       # EMA, RSI, MACD, Bollinger, ATR
│   ├── regime_detector.py  # Bull / Bear / Sideways / High-Vol
│   ├── signals.py          # Legacy indicator-based signals
│   ├── risk_manager.py     # ATR-based sizing (legacy)
│   ├── paper_trader.py     # Long position paper engine
│   ├── short_tracker.py    # Simulated short position tracker
│   └── performance.py      # Sharpe, Sortino, win rate
│
├── quant/                  # Institutional quant layer
│   ├── features.py         # Returns, momentum, z-score, vol, volume spike
│   ├── alpha.py            # 4-factor alpha signal extraction
│   ├── ensemble.py         # Weighted linear combiner → composite score
│   ├── portfolio.py        # Asset ranking + correlation filter + long/short selection
│   └── risk.py             # Volatility-adjusted position sizing + kill-switch
│
├── strategies/
│   ├── trend_following.py  # EMA crossover + momentum
│   ├── mean_reversion.py   # RSI + Bollinger Band reversion
│   └── breakout.py         # Volume breakout detection
│
├── backtest/
│   └── engine.py           # Walk-forward backtesting framework
│
├── dashboard/
│   ├── api.py              # FastAPI REST dashboard
│   └── index.html          # Web UI (auto-refreshes every 30s)
│
├── config.py               # All configuration constants
└── main.py                 # Main bot loop (QUANT_MODE toggle)
```

---

## Alpha Factor Pipeline

### 1. Feature Engineering (`quant/features.py`)
- `pct_return_1d`, `pct_return_5d` — raw returns
- `momentum` — rolling return normalised via tanh
- `mean_reversion` — z-score vs rolling mean, inverted
- `volatility_signal` — normalised rolling std (ranked vs 60-period history)
- `volume_spike` — relative volume vs 20-period average

### 2. Alpha Signals (`quant/alpha.py`)
Each signal normalised to `[-1, +1]`:

| Factor | Weight | Description |
|--------|--------|-------------|
| Momentum | 35% | Time-series momentum |
| Mean Reversion | 25% | Z-score reversion signal |
| Volume | 25% | Volume spike confirmation |
| Volatility | 15% | Regime filter |

### 3. Ensemble Score (`quant/ensemble.py`)
```
score = 0.35 × momentum + 0.25 × mean_rev + 0.25 × volume + 0.15 × volatility
```
Score in `[-1, +1]` — positive = long candidate, negative = short candidate.

### 4. Portfolio Construction (`quant/portfolio.py`)
- Rank all assets by ensemble score
- Select top-N as **LONG**, bottom-N as **SHORT**
- Correlation filter: exclude assets with `r > 0.80` vs existing positions

### 5. Risk Management (`quant/risk.py`)
```
position_size = (capital × risk_per_trade) / volatility
```
- Min: $10 · Max: $200 per position
- 20% capital always in reserve
- 15% portfolio drawdown kill-switch

---

## Asset Universe

### Crypto (KuCoin)
`BTC/USDT` · `ETH/USDT` · `SOL/USDT` · `BNB/USDT` · `AVAX/USDT` · `LINK/USDT`

### Equities (Alpaca Paper)
`NVDA` · `TSLA` · `AAPL` · `MSFT` · `AMZN` · `SPY`

---

## Running the Bot

```bash
# Clone
git clone https://github.com/Bernard-Baah/midastouch.git
cd midastouch

# Install
pip install -r requirements.txt

# Configure (create .env file)
echo "ALPACA_API_KEY=your_key" >> .env
echo "ALPACA_SECRET=your_secret" >> .env
echo "ALPACA_BASE_URL=https://paper-api.alpaca.markets" >> .env

# Run bot
python main.py

# Run dashboard (separate terminal)
uvicorn dashboard.api:app --host 0.0.0.0 --port 8000

# Run backtest
python backtest/engine.py
```

---

## Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web UI dashboard |
| `GET /status` | Capital, return, drawdown |
| `GET /positions` | Open long positions |
| `GET /trades?limit=50` | Full trade history |
| `GET /performance` | Sharpe, Sortino, win rate, P&L |
| `GET /signals` | Current signals per symbol |
| `GET /health` | Health check |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QUANT_MODE` | `True` | Use quant pipeline (False = legacy indicators) |
| `QUANT_TOP_N` | `2` | Max simultaneous long positions |
| `QUANT_BOTTOM_N` | `1` | Max simultaneous short positions |
| `INITIAL_CAPITAL` | `$1,000` | Starting paper capital |
| `MAX_DRAWDOWN_LIMIT` | `15%` | Portfolio kill-switch |
| `RESERVE_PCT` | `20%` | Minimum cash reserve |
| `LOOP_INTERVAL_SECONDS` | `300` | Bot loop frequency (5 min) |

---

## Roadmap

- [x] Phase 1: Core paper trading engine (RSI, MACD, EMA)
- [x] Phase 2: Quant layer — multi-factor alpha + ensemble scoring
- [x] Phase 3: Portfolio construction + correlation filter
- [x] Phase 4: Short selling (paper simulation)
- [x] Phase 5: Alpaca stock integration (NVDA, TSLA, AAPL, MSFT, AMZN, SPY)
- [ ] Phase 6: Reinforcement learning + online model adaptation
- [ ] Phase 7: Live trading (Binance account + Alpaca live)
- [ ] Phase 8: Options strategies

---

## Disclaimer

This system is for **paper trading and educational purposes only**.
Past performance does not guarantee future results.
Never trade with money you cannot afford to lose.
