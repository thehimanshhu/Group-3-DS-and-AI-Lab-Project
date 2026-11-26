from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from src.metrics import eer

runs = ["F-01", "F-02", "F-03", "D-01", "A-01", "A-02"]
splits = [
    "valid", "test_indomain", "test_unseen_gen",
    "test_phone", "test_itw", "test_asvspoof",
]
n_bins = 15
min_temp, max_temp = 0.05, 20.0
plot_dir = Path("assets/infographics/Milestone-5")
plot_dir.mkdir(parents=True, exist_ok=True)


def reliability(y, probabilities):
    bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    occupied = counts > 0
    weights = counts[occupied]
    predicted = np.bincount(
        bins, weights=probabilities, minlength=n_bins
    )[occupied] / weights
    observed = np.bincount(
        bins, weights=y, minlength=n_bins
    )[occupied] / weights
    ece = np.average(np.abs(predicted - observed), weights=weights)
    return predicted, observed, ece


figures = {
    split: plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for split in splits
}
results = []

for index, run in enumerate(runs):
    path = Path("results") / run / "predictions_full.csv.gz"
    frame = pd.read_csv(path, usecols=["split", "label", "p_fake"])
    y = frame["label"].eq("fake").to_numpy()
    scores = frame["p_fake"].to_numpy()
    logits = logit(np.clip(scores, 1e-6, 1 - 1e-6))
    valid = frame["split"].eq("valid").to_numpy()
    _, cutoff = eer(y[valid], scores[valid])

    def validation_loss(log_temperature):
        scaled = logits[valid] / np.exp(log_temperature)
        return np.logaddexp(0, np.where(y[valid], -scaled, scaled)).mean()

    fit = minimize_scalar(
        validation_loss, bounds=np.log([min_temp, max_temp]), method="bounded"
    )
    assert fit.success, f"Temperature fit failed for {run}"
    temperature = np.exp(fit.x)
    scaled_logits = logits / temperature
    calibrated = expit(scaled_logits)
    cutoff_logit = logit(np.clip(cutoff, 1e-6, 1 - 1e-6)) / temperature
    raw_decisions = scores >= cutoff
    calibrated_decisions = scaled_logits >= cutoff_logit
    assert np.array_equal(raw_decisions, calibrated_decisions), run

    for split in splits:
        selected = frame["split"].eq(split).to_numpy()
        fakes = selected & y
        ax = figures[split][1].flat[index]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

        row = {
            "run": run,
            "split": split,
            "temperature": temperature,
            "temperature_at_search_limit": (
                np.isclose(temperature, min_temp, rtol=1e-3)
                or np.isclose(temperature, max_temp, rtol=1e-3)
            ),
            "fake_recall_before_pct": 100 * raw_decisions[fakes].mean(),
            "fake_recall_after_pct": 100 * calibrated_decisions[fakes].mean(),
        }
        for stage, probabilities in [("before", scores), ("after", calibrated)]:
            predicted, observed, ece = reliability(y[selected], probabilities[selected])
            row[f"ece_{stage}"] = ece
            ax.plot(predicted, observed, "o-", label=f"{stage.title()} ({ece:.3f})")

        ax.set(
            title=run, xlim=(0, 1), ylim=(0, 1),
            xlabel="Mean predicted fake probability", ylabel="Observed fake rate",
        )
        ax.legend(fontsize=8)
        results.append(row)

for split, (fig, _) in figures.items():
    fig.suptitle(f"Reliability diagram — {split} — legend shows ECE")
    output = plot_dir / f"{split}_reliability.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)

output = Path("results/milestone5_calibration.csv")
pd.DataFrame(results).to_csv(output, index=False, float_format="%.6f")
print(f"Saved {len(results)} rows to {output}")
