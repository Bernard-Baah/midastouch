"""
MidasTouch - Ensemble Signal Combiner
Aggregates signals from all three strategies, weighted by market regime.
Only recommends trades when the ensemble score clears MIN_SIGNAL_STRENGTH.
"""

import logging
from typing import Dict, Tuple

import pandas as pd

from config import MIN_SIGNAL_STRENGTH, REGIME_WEIGHTS
import strategies.trend_following as trend_following
import strategies.mean_reversion  as mean_reversion
import strategies.breakout        as breakout

logger = logging.getLogger(__name__)


def generate_ensemble_signal(
    df: pd.DataFrame,
    regime: str,
) -> Dict:
    """
    Generate a combined trading signal from all three strategies.

    Each strategy score is weighted according to the current market regime
    (see config.REGIME_WEIGHTS). The weighted average becomes the ensemble score.
    A direction is returned only when |score| >= MIN_SIGNAL_STRENGTH.

    Args:
        df:     DataFrame with indicator columns already calculated.
        regime: Current market regime string (e.g. 'bull', 'bear', 'sideways')

    Returns:
        Dict with keys:
            score       (float -1.0 to +1.0)
            direction   ('buy', 'sell', 'hold')
            regime      (str — echoed back for convenience)
            components  (dict of per-strategy raw scores)
            weights     (dict of per-strategy weights used)
            actionable  (bool — True when |score| >= MIN_SIGNAL_STRENGTH)
    """
    # ── Get raw signals from each strategy ────────────────────────────────────
    trend_score, _  = trend_following.generate_signal(df)
    mr_score, _     = mean_reversion.generate_signal(df)
    bo_score, _     = breakout.generate_signal(df)

    components = {
        'trend_following': trend_score,
        'mean_reversion':  mr_score,
        'breakout':        bo_score,
    }

    # ── Get regime-specific weights ────────────────────────────────────────────
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['sideways'])

    # ── Weighted ensemble score ────────────────────────────────────────────────
    ensemble_score = (
        trend_score * weights['trend_following'] +
        mr_score    * weights['mean_reversion']  +
        bo_score    * weights['breakout']
    )
    ensemble_score = max(-1.0, min(1.0, ensemble_score))

    # ── Determine direction based on threshold ─────────────────────────────────
    actionable = abs(ensemble_score) >= MIN_SIGNAL_STRENGTH
    if not actionable:
        direction = 'hold'
    elif ensemble_score > 0:
        direction = 'buy'
    else:
        direction = 'sell'

    logger.info(
        "Ensemble signal: score=%.3f direction=%s regime=%s actionable=%s | "
        "trend=%.3f mr=%.3f bo=%.3f",
        ensemble_score, direction, regime, actionable,
        trend_score, mr_score, bo_score,
    )

    return {
        'score':      ensemble_score,
        'direction':  direction,
        'regime':     regime,
        'components': components,
        'weights':    weights,
        'actionable': actionable,
    }


def generate_all_symbols(
    symbol_data: Dict[str, pd.DataFrame],
    symbol_regimes: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Generate ensemble signals for multiple symbols at once.

    Args:
        symbol_data:    Dict mapping symbol → indicator DataFrame
        symbol_regimes: Dict mapping symbol → regime string

    Returns:
        Dict mapping symbol → signal dict (same structure as generate_ensemble_signal)
    """
    results = {}
    for symbol, df in symbol_data.items():
        regime = symbol_regimes.get(symbol, 'sideways')
        try:
            results[symbol] = generate_ensemble_signal(df, regime)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Signal generation failed for %s: %s", symbol, exc)
            results[symbol] = {
                'score': 0.0, 'direction': 'hold', 'regime': regime,
                'components': {}, 'weights': {}, 'actionable': False,
            }
    return results

# Alias for compatibility
get_ensemble_signal = generate_ensemble_signal

