"""
MidasTouch - Data Feed Module
Fetches OHLCV market data from Binance via ccxt.
Supports multi-timeframe fetching with in-memory caching.
"""

import time
import logging
from typing import Dict, Optional

import ccxt
import pandas as pd

from config import (
    EXCHANGE_ID, EXCHANGE_RATE_LIMIT, SYMBOLS, TIMEFRAMES,
    OHLCV_LIMIT, CACHE_TTL_SECONDS
)

logger = logging.getLogger(__name__)


class DataFeed:
    """
    Fetches and caches OHLCV data from Binance (public endpoint).
    No API key required for market data.
    """

    def __init__(self):
        """Initialise the ccxt Binance exchange connector and cache."""
        self.exchange = ccxt.binance({
            'enableRateLimit': EXCHANGE_RATE_LIMIT,
            'options': {'defaultType': 'spot'},
        })
        # Cache: {(symbol, timeframe): {'df': DataFrame, 'ts': timestamp}}
        self._cache: Dict[tuple, dict] = {}
        logger.info("DataFeed initialised — exchange: %s", self.exchange.id)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = OHLCV_LIMIT,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles for a symbol/timeframe pair.

        Returns a DataFrame with columns:
            timestamp, open, high, low, close, volume
        indexed by timestamp (UTC datetime).

        Uses an in-memory cache; staleness determined by CACHE_TTL_SECONDS.

        Args:
            symbol:        Trading pair, e.g. 'BTC/USDT'
            timeframe:     Candle timeframe, e.g. '1h'
            limit:         Number of candles to fetch
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            pd.DataFrame or None on error
        """
        cache_key = (symbol, timeframe)
        ttl = CACHE_TTL_SECONDS.get(timeframe, 60)

        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            age = time.time() - cached['ts']
            if age < ttl:
                logger.debug(
                    "Cache hit for %s/%s (age=%.0fs)", symbol, timeframe, age
                )
                return cached['df']

        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except ccxt.NetworkError as exc:
            logger.error("Network error fetching %s/%s: %s", symbol, timeframe, exc)
            return self._cache.get(cache_key, {}).get('df')
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error fetching %s/%s: %s", symbol, timeframe, exc)
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Unexpected error fetching %s/%s: %s", symbol, timeframe, exc)
            return None

        if not raw:
            logger.warning("Empty response for %s/%s", symbol, timeframe)
            return None

        df = pd.DataFrame(
            raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        df.sort_index(inplace=True)

        self._cache[cache_key] = {'df': df, 'ts': time.time()}
        logger.debug(
            "Fetched %d candles for %s/%s", len(df), symbol, timeframe
        )
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: list = None,
        limit: int = OHLCV_LIMIT,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch OHLCV data for multiple timeframes for a single symbol.

        Args:
            symbol:     Trading pair
            timeframes: List of timeframe strings; defaults to config.TIMEFRAMES
            limit:      Number of candles per timeframe

        Returns:
            Dict mapping timeframe → DataFrame (or None on error)
        """
        timeframes = timeframes or TIMEFRAMES
        result: Dict[str, Optional[pd.DataFrame]] = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, limit=limit)
        return result

    def fetch_all_symbols(
        self,
        timeframe: str = '1h',
        symbols: list = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch OHLCV data for all configured symbols on a single timeframe.

        Args:
            timeframe: Candle timeframe
            symbols:   List of symbols; defaults to config.SYMBOLS

        Returns:
            Dict mapping symbol → DataFrame
        """
        symbols = symbols or SYMBOLS
        result: Dict[str, Optional[pd.DataFrame]] = {}
        for sym in symbols:
            result[sym] = self.fetch_ohlcv(sym, timeframe)
        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Return the most recent close price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            float price or None on error
        """
        df = self.fetch_ohlcv(symbol, '1h', limit=1)
        if df is not None and not df.empty:
            return float(df['close'].iloc[-1])
        return None

    def clear_cache(self, symbol: str = None, timeframe: str = None) -> None:
        """
        Invalidate cache entries.

        Args:
            symbol:    If given, clear only entries for this symbol
            timeframe: If given alongside symbol, clear that specific pair
        """
        if symbol and timeframe:
            self._cache.pop((symbol, timeframe), None)
        elif symbol:
            keys_to_remove = [k for k in self._cache if k[0] == symbol]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            self._cache.clear()
        logger.debug("Cache cleared (symbol=%s, tf=%s)", symbol, timeframe)
