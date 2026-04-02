"""
MidasTouch — Stock Data Feed (Alpaca Paper Trading)
Fetches OHLCV bars for equities via Alpaca Markets REST API.
Credentials loaded from .env file.
"""

import logging
import os
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY  = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET   = os.getenv("ALPACA_SECRET", "")
DATA_BASE_URL   = "https://data.alpaca.markets"

TIMEFRAME_MAP = {
    "15m": "15Min",
    "1h":  "1Hour",
    "4h":  "4Hour",
    "1d":  "1Day",
}


class StockFeed:
    """Fetches OHLCV bars from Alpaca for equity symbols."""

    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        self._cache = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV bars for an equity symbol.

        Args:
            symbol:    Ticker e.g. 'AAPL'
            timeframe: '15m', '1h', '4h', '1d'
            limit:     Number of bars to fetch

        Returns:
            DataFrame with columns: open, high, low, close, volume
            or None on error.
        """
        tf = TIMEFRAME_MAP.get(timeframe, "1Hour")
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=limit * _timeframe_hours(timeframe))

        url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": tf,
            "start":     start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":       end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":     limit,
            "feed":      "iex",
        }

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            resp.raise_for_status()
            bars = resp.json().get("bars", [])
            if not bars:
                logger.warning("No bars returned for %s", symbol)
                return None

            df = pd.DataFrame(bars)
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            df = df.sort_index()
            logger.info("Fetched %d bars for %s (%s)", len(df), symbol, timeframe)
            return df

        except Exception as exc:
            logger.error("StockFeed error for %s: %s", symbol, exc)
            return None

    def get_account(self) -> dict:
        """Fetch Alpaca paper account info."""
        try:
            resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Account fetch error: %s", exc)
            return {}


def _timeframe_hours(tf: str) -> float:
    return {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}.get(tf, 1.0)
