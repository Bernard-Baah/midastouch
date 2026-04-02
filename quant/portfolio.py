"""
MidasTouch Quant Layer — Portfolio Construction
Ranks assets by ensemble score and selects longs/shorts.
Also implements a correlation filter to prevent adding
highly correlated positions.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configurable thresholds
LONG_THRESHOLD  =  0.05   # Score above this → long candidate
SHORT_THRESHOLD = -0.05   # Score below this → short candidate
MAX_CORRELATION =  0.80   # Reject asset if correlated > this with existing longs
TOP_N           =  2      # Max simultaneous longs
BOTTOM_N        =  2      # Max simultaneous shorts


def rank_assets(scores: dict[str, float]) -> pd.DataFrame:
    """
    Convert a {symbol: score} dict to a sorted DataFrame.

    Returns DataFrame with columns [symbol, score] sorted descending.
    """
    if not scores:
        return pd.DataFrame(columns=["symbol", "score"])

    df = pd.DataFrame(list(scores.items()), columns=["symbol", "score"])
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def select_portfolio(
    scores: dict[str, float],
    price_history: Optional[dict[str, pd.Series]] = None,
    top_n: int = TOP_N,
    bottom_n: int = BOTTOM_N,
) -> tuple[list[str], list[str]]:
    """
    Select long and short candidates from ranked scores.

    Args:
        scores:        {symbol: ensemble_score}
        price_history: {symbol: pd.Series of close prices} for correlation filter
        top_n:         Max long positions
        bottom_n:      Max short positions

    Returns:
        (longs, shorts) — lists of symbol strings
    """
    ranked = rank_assets(scores)
    if ranked.empty:
        return [], []

    long_candidates  = ranked[ranked["score"] >= LONG_THRESHOLD]["symbol"].tolist()
    short_candidates = ranked[ranked["score"] <= SHORT_THRESHOLD]["symbol"].tolist()
    short_candidates.reverse()   # worst first

    longs  = _apply_correlation_filter(long_candidates[:top_n],   price_history)
    shorts = _apply_correlation_filter(short_candidates[:bottom_n], price_history)

    logger.info(
        "Portfolio | longs=%s shorts=%s (scores: %s)",
        longs, shorts,
        {s: round(v, 3) for s, v in scores.items()}
    )
    return longs, shorts


def _apply_correlation_filter(
    candidates: list[str],
    price_history: Optional[dict[str, pd.Series]],
) -> list[str]:
    """
    Remove assets whose returns are correlated > MAX_CORRELATION
    with any already-selected asset.
    Returns filtered list preserving rank order.
    """
    if price_history is None or len(candidates) <= 1:
        return candidates

    selected   = [candidates[0]]
    returns    = {
        sym: price_history[sym].pct_change().dropna()
        for sym in candidates
        if sym in price_history and len(price_history[sym]) > 5
    }

    for candidate in candidates[1:]:
        if candidate not in returns:
            selected.append(candidate)
            continue

        correlated = False
        for sel in selected:
            if sel not in returns:
                continue
            # Align series before computing correlation
            a, b = returns[sel].align(returns[candidate], join="inner")
            if len(a) < 5:
                continue
            corr = a.corr(b)
            if not np.isnan(corr) and abs(corr) > MAX_CORRELATION:
                logger.debug(
                    "Correlation filter: %s vs %s = %.2f (excluded)", candidate, sel, corr
                )
                correlated = True
                break

        if not correlated:
            selected.append(candidate)

    return selected
