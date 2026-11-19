import numpy as np
from sklearn.metrics import roc_curve


def eer(is_fake, scores):
    """Return (equal error rate as a fraction, score threshold)."""
    labels = np.asarray(is_fake)
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("EER requires both real (0) and fake (1) labels.")

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Skip the synthetic threshold above every score.
    index = int(np.argmin(np.abs(fpr[1:] - fnr[1:]))) + 1
    return float((fpr[index] + fnr[index]) / 2), float(thresholds[index])