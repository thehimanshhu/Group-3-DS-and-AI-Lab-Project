"""Regenerate Milestone 5 evaluation tables and figures from saved scores."""

from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_languages, evaluate_splits
from src.metrics import eer

RESULTS = ROOT / "results"
PLOTS = ROOT / "assets/infographics/Milestone-5"
RUNS = ("F-01", "F-02", "F-03", "D-01", "A-01", "A-02")
SPLITS = {
    "test_indomain": "In-domain",
    "test_unseen_gen": "Unseen generators",
    "test_phone": "Phone channel",
    "test_itw": "In-the-Wild",
    "test_asvspoof": "ASVspoof eval",
}
KEY = ["split", "clip_id", "label"]


def load_predictions():
    predictions = {}
    reference = None

    for run in RUNS:
        path = RESULTS / run / "predictions_full.csv.gz"
        frame = pd.read_csv(path, usecols=KEY + ["p_fake", "language", "generator"])
        test = frame.loc[frame["split"].isin(SPLITS), KEY]
        cohort = test.sort_values(KEY).reset_index(drop=True)

        if reference is None:
            reference = cohort
        else:
            pd.testing.assert_frame_equal(reference, cohort, obj=f"{run} test clips")

        predictions[run] = frame

    return predictions


def bootstrap_eer(predictions):
    rng = np.random.default_rng(42)
    records = []

    for run, frame in predictions.items():
        for split in SPLITS:
            rows = frame.loc[frame["split"].eq(split)]
            real = rows.loc[rows["label"].eq("real"), "p_fake"].to_numpy()
            fake = rows.loc[rows["label"].eq("fake"), "p_fake"].to_numpy()
            if not real.size or not fake.size or real.size + fake.size != len(rows):
                raise ValueError(f"{run} {split} needs both classes and valid labels")

            labels = np.repeat([0, 1], [real.size, fake.size])
            estimate, _ = eer(labels, np.concatenate([real, fake]))

            samples = np.empty(1_000)
            for i in range(samples.size):
                scores = np.concatenate([
                    rng.choice(real, size=real.size, replace=True),
                    rng.choice(fake, size=fake.size, replace=True),
                ])
                samples[i], _ = eer(labels, scores)

            lower, upper = np.percentile(samples, [2.5, 97.5])
            records.append({
                "run": run,
                "split": split,
                "clips": len(rows),
                "eer_pct": 100 * estimate,
                "ci_low_pct": 100 * lower,
                "ci_high_pct": 100 * upper,
            })
            print(f"Bootstrapped {run} {split}", flush=True)

    return pd.DataFrame(records)


def breakdowns(predictions, reports):
    group_rows = []
    language_reports = []

    for run, frame in predictions.items():
        language_report = evaluate_languages(frame)
        language_report.insert(0, "run", run)
        language_reports.append(language_report)
        cutoff = reports[run].loc["test_unseen_gen", "threshold"]

        for split in ("test_unseen_gen", "test_asvspoof"):
            test = frame.loc[frame["split"].eq(split)]
            real = test.loc[test["label"].eq("real")]
            fake = test.loc[test["label"].eq("fake")]

            for (language, generator), group in fake.groupby(["language", "generator"]):
                matched_real = real.loc[real["language"].eq(language)]
                comparison = pd.concat([matched_real, group], ignore_index=True)
                group_eer, _ = eer(comparison["label"].eq("fake"), comparison["p_fake"])
                caught = int(group["p_fake"].ge(cutoff).sum())

                group_rows.append({
                    "run": run,
                    "split": split,
                    "language": language,
                    "generator_or_attack": generator,
                    "real_clips": len(matched_real),
                    "fake_clips": len(group),
                    "fakes_caught": caught,
                    "fake_recall_pct": 100 * caught / len(group),
                    "eer_pct": 100 * group_eer,
                    "validation_cutoff": cutoff,
                })

    groups = pd.DataFrame(group_rows)
    return {
        "by_generator": groups.loc[groups["split"].eq("test_unseen_gen")],
        "by_attack": groups.loc[groups["split"].eq("test_asvspoof")],
        "by_language": pd.concat(language_reports, ignore_index=True),
    }


def save_table(name, frame):
    path = RESULTS / f"milestone5_{name}.csv"
    frame.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved {len(frame)} rows to {path.relative_to(ROOT)}", flush=True)


def save_plot(fig, filename):
    fig.savefig(PLOTS / filename, dpi=150)
    plt.close(fig)
    print((PLOTS / filename).relative_to(ROOT), flush=True)


def plot_test_splits(predictions, reports):
    bins = np.linspace(0, 1, 41)

    for split, title in SPLITS.items():
        curves, (roc_ax, pr_ax, det_ax) = plt.subplots(
            1, 3, figsize=(16, 5), layout="constrained"
        )
        confusion, confusion_axes = plt.subplots(
            2, 3, figsize=(12, 7), layout="constrained"
        )
        distributions, distribution_axes = plt.subplots(
            2, 3, figsize=(13, 7), sharex=True, sharey=True, layout="constrained"
        )

        for run, cm_ax, score_ax in zip(
            RUNS, confusion_axes.flat, distribution_axes.flat
        ):
            frame = predictions[run]
            rows = frame.loc[frame["split"].eq(split)]
            is_fake = rows["label"].eq("fake")
            scores = rows["p_fake"]

            RocCurveDisplay.from_predictions(is_fake, scores, name=run, ax=roc_ax)
            PrecisionRecallDisplay.from_predictions(is_fake, scores, name=run, ax=pr_ax)
            DetCurveDisplay.from_predictions(is_fake, scores, name=run, ax=det_ax)

            row = reports[run].loc[split]
            counts = np.array([
                [row["real_as_real"], row["real_as_fake"]],
                [row["fake_as_real"], row["fake_as_fake"]],
            ], dtype=int)
            if counts.sum() != row["clips"]:
                raise ValueError(f"Confusion counts do not match clip count for {run} {split}")

            ConfusionMatrixDisplay(counts, display_labels=["Real", "Fake"]).plot(
                ax=cm_ax, cmap="Blues", colorbar=False, values_format=",d"
            )
            cutoff = row["threshold"]
            cm_ax.set_title(f"{run} · cutoff {cutoff:.5f}")

            for label, color in (("real", "tab:blue"), ("fake", "tab:orange")):
                class_scores = scores.loc[rows["label"].eq(label)]
                hist, _ = np.histogram(class_scores, bins=bins)
                score_ax.stairs(
                    100 * hist / len(class_scores), bins,
                    label=label.title(), color=color,
                )

            score_ax.axvline(
                cutoff, color="black", linestyle="--", label="Validation cutoff"
            )
            score_ax.set(title=run, xlim=(0, 1))
            score_ax.legend(fontsize=8)

        roc_ax.set_title("ROC")
        pr_ax.set_title("Precision–Recall")
        det_ax.set_title("DET")
        curves.suptitle(f"{title} — full test set")
        confusion.suptitle(f"{split} — confusion matrices at validation cutoffs")
        distributions.suptitle(f"{split} — fake-score distributions")
        distributions.supxlabel("Predicted fake score")
        distributions.supylabel("Clips per bin (% of each class)")

        save_plot(curves, f"{split}_curves.png")
        save_plot(confusion, f"{split}_confusion.png")
        save_plot(distributions, f"{split}_score_distribution.png")


def main():
    predictions = load_predictions()
    reports = {
        run: evaluate_splits(frame).set_index("split").loc[list(SPLITS)]
        for run, frame in predictions.items()
    }

    intervals = bootstrap_eer(predictions)
    metrics = pd.concat(reports, names=["run"]).reset_index()
    comparison = metrics.merge(
        intervals,
        on=["run", "split", "clips"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_bootstrap"),
    )
    if len(comparison) != 30 or comparison[["ci_low_pct", "ci_high_pct"]].isna().any().any():
        raise ValueError("Expected one bootstrap interval for each model and test split")
    if not np.allclose(comparison["eer_pct"], comparison["eer_pct_bootstrap"], atol=1e-6):
        raise ValueError("Comparison and bootstrap EER estimates differ")

    save_table("eer_bootstrap", intervals)
    save_table("model_comparison", comparison.drop(columns="eer_pct_bootstrap"))
    for name, table in breakdowns(predictions, reports).items():
        save_table(name, table)

    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_test_splits(predictions, reports)
    subprocess.run([sys.executable, "-m", "src.calibrate"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()