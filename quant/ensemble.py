"""
MidasTouch Quant Layer — Ensemble Scorer
Combines multi-factor alpha signals into a single composite score.
Weights are configurable via FACTOR_WEIGHTS in alpha.py.
"""

import logging
import numpy as np
from quant.alpha import FACTOR_WEIGHTS

logger = logging.getLogger(__name__)


def compute_ensemble_score(signals: dict[str, float]) -> float:
    """
    Weighted linear combination of alpha signals.

    score = sum(weight_i * signal_i) / sum(weights)

    Returns:
        float in [-1, +1]
        Positive → bullish (long candidate)
        Negative → bearish (short candidate)
    """
    if not signals:
        return 0.0

    total_weight = sum(FACTOR_WEIGHTS.values())
    if total_weight == 0:
        return 0.0

    # Map signal keys to weight keys
    key_map = {
        "momentum":   "momentum",
        "mean_rev":   "mean_rev",
        "volatility": "volatility",
        "volume":     "volume",
    }

    score = 0.0
    for sig_key, weight_key in key_map.items():
        weight = FACTOR_WEIGHTS.get(weight_key, 0.0)
        value  = signals.get(sig_key, 0.0)
        if np.isnan(value):
            value = 0.0
        score += weight * value

    score /= total_weight
    score  = float(np.clip(score, -1.0, 1.0))

    logger.debug("Ensemble score: %.4f (raw signals: %s)", score, signals)
    return score
