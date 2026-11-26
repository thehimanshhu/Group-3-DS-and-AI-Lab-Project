"""Compare two runs on the same clips: paired bootstrap of the EER difference, and Holm's correction."""
import numpy as np

from src.metrics import eer


def paired_bootstrap(is_fake, scores_a, scores_b, resamples=2000, seed=0):
    """EER(b) - EER(a) on the clips, and its bootstrap distribution (same resampled clips for both runs).

    Returns (difference, resampled differences). Negative means b has the lower EER.
    """
    y = np.asarray(is_fake, dtype=int)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        while y[idx].min() == y[idx].max():  # both classes needed for an EER
            idx = rng.integers(0, len(y), len(y))
        diffs[i] = eer(y[idx], b[idx])[0] - eer(y[idx], a[idx])[0]
    return eer(y, b)[0] - eer(y, a)[0], diffs


def interval(diffs, level=0.95):
    """Percentile interval of the resampled differences."""
    tail = (1 - level) / 2 * 100
    return tuple(np.percentile(diffs, [tail, 100 - tail]))


def p_value(diffs):
    """Two-sided bootstrap p-value for 'no difference': twice the share of resamples on the far side of 0."""
    diffs = np.asarray(diffs)
    share = min((diffs >= 0).mean(), (diffs <= 0).mean())
    return min(1.0, 2 * max(share, 1 / len(diffs)))  # never below the resolution of the bootstrap


def holm(p_values, alpha=0.05):
    """Holm's step-down correction: (adjusted p-values, rejected) in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted, adjusted < alpha
