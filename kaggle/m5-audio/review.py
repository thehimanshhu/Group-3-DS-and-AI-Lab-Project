"""Private F-03 playback, saliency, and full-test quiet-frame analysis."""

import base64
import hashlib
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import soundfile as sf

SCORER = "1756e7e4b9a1f0320e664c34f0901648f5fc0880"
MANIFEST_SHA = "35826f269b44f83afd914342c16190f2f69c75f0127af8fbf2475bd9995412c8"
PREDICTIONS_SHA = "a3fe6aea4e36aa4bfe37acce666302893a09fc1d6c9e880028117cdb21df05a0"
KEY = ["split", "clip_id"]
BANDS = ["< -30 dB", "-30 to < -20 dB", "-20 to < -10 dB", ">= -10 dB"]


def find_input(root, name):
    """The single mounted folder called name, in either Kaggle layout."""
    matches = (
        sorted(root.glob(f"datasets/*/{name}"))
        + sorted(root.glob(f"notebooks/*/{name}"))
        + sorted(root.glob(name))
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one mounted {name} under {root}; found {len(matches)}."
        )
    return matches[0]


def select_cases(rows):
    """Take the strongest mistake per split, then three of each error type."""
    selected = []
    seen = set()
    for label in ("real", "fake"):
        errors = rows.loc[rows["label"].eq(label) & rows["wrong"]]
        errors = errors.sort_values(
            ["p_fake", "clip_id"], ascending=[label == "fake", True]
        )
        splits = set()
        for row in errors.to_dict("records"):
            utterance = row["clip_id"].removeprefix("phone_")
            if utterance in seen or row["split"] in splits:
                continue
            splits.add(row["split"])
            seen.add(utterance)
            selected.append(row)
            if len(splits) == 3:
                break
        if len(splits) != 3:
            raise ValueError(f"Could not find three distinct review cases for {label}.")
    return pd.DataFrame(selected).assign(case=[f"C{i:02}" for i in range(1, 7)])


def quiet_frame_db(wave, sample_rate):
    """10 log10(p10 of 20 ms frame power / mean frame power), on processed audio."""
    width = sample_rate // 50
    frames = wave[: len(wave) // width * width].reshape(-1, width)
    power = np.mean(frames * frames, axis=1)
    if not len(power) or power.mean() <= 0:
        return np.nan
    return float(10 * np.log10(max(np.quantile(power, 0.10) / power.mean(), 1e-12)))


def read_audio(path, sample_rate):
    wave, rate = sf.read(path, dtype="float32")
    if rate != sample_rate or wave.ndim != 1:
        raise ValueError(f"Expected mono {sample_rate} Hz audio: {path}")
    return wave


def summarize_quiet(rows):
    bins = pd.cut(
        rows["quiet_frame_db"],
        [-np.inf, -30, -20, -10, np.inf],
        labels=BANDS,
        right=False,
    )
    bands = bins.cat.add_categories(["undefined"]).fillna("undefined")
    summary = rows.assign(band=bands).groupby(["split", "label", "band"], observed=True)
    summary = summary.agg(
        clips=("wrong", "size"), errors=("wrong", "sum")
    ).reset_index()
    return summary.assign(error_pct=100 * summary["errors"] / summary["clips"])


def review_artifacts(cases, out, model, device, data_cfg, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.explain import saliency

    sample_rate = data_cfg["sample_rate"]
    articles, observations = [], []
    for row in cases.itertuples(index=False):
        audio = read_audio(row.path, sample_rate)
        intensity = saliency(
            model, audio, device, sample_rate, data_cfg["crop_seconds"]
        )
        if intensity.shape != audio.shape or not np.isfinite(intensity).all():
            raise ValueError(f"Unexpected saliency values for {row.case}.")
        times = np.arange(len(audio)) / sample_rate
        fig, axes = plt.subplots(
            2, 1, figsize=(9, 3.8), sharex=True, layout="constrained"
        )
        axes[0].plot(times, audio, color="#34465a", linewidth=0.5)
        axes[0].set_ylabel("Waveform")
        axes[1].plot(times, intensity, color="#b75223", linewidth=0.9)
        axes[1].set(ylabel="Gradient magnitude", xlabel="Time (s)")
        image = io.BytesIO()
        fig.savefig(image, format="png", dpi=115)
        plt.close(fig)
        png = base64.b64encode(image.getvalue()).decode("ascii")
        wav = base64.b64encode(Path(row.path).read_bytes()).decode("ascii")
        verdict = "fake" if row.label == "real" else "real"
        description = (
            f"{row.split} · {row.language} · {row.generator} | true: {row.label} | "
            f"score: {row.p_fake:.5f} | verdict: {verdict}"
        )
        articles.append(
            f"<section><h2>{row.case}</h2><p>{escape(description)}</p>"
            f'<audio controls preload="none" src="data:audio/wav;base64,{wav}"></audio>'
            f'<img alt="Waveform and saliency for {row.case}" src="data:image/png;base64,{png}"></section>'
        )
        total = intensity.sum(dtype=np.float64)
        observations.append(
            {
                "case": row.case,
                "split": row.split,
                "clip_id": row.clip_id,
                "language": row.language,
                "generator": row.generator,
                "label": row.label,
                "p_fake": row.p_fake,
                "threshold": threshold,
                "verdict": verdict,
                "duration_s": len(audio) / sample_rate,
                "saliency_peak_s": np.argmax(intensity) / sample_rate
                if total
                else np.nan,
                "saliency_first_second_pct": 100 * intensity[:sample_rate].sum() / total
                if total
                else np.nan,
            }
        )
    (out / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Private F-03 review</title>'
        "<style>body{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}"
        "img{display:block;max-width:100%}section{border-top:1px solid #ccc;margin:1rem 0}</style>"
        "<h1>Private F-03 listening review</h1>"
        "<p>For each clip, note a timestamp and what you hear, or write “nothing distinctive.” "
        "The plot shows gradient magnitude, not why the model made the mistake. Keep this page private.</p>"
        + "".join(articles)
        + "</html>",
        encoding="utf-8",
    )
    pd.DataFrame(observations).to_csv(out / "review_cases.csv", index=False)


def main():
    import torch

    inputs = Path("/kaggle/input")
    data = find_input(inputs, "dsai-group3-m4-data")
    english = find_input(inputs, "dsai-group3-m5-full-tests-en")
    score_root = find_input(inputs, "dsai-group3-m5-score-vikram")
    (prediction_path,) = score_root.rglob("predictions_full.csv.gz")
    manifest_path = data / "manifests/manifest_m4.csv"
    for path, expected in (
        (manifest_path, MANIFEST_SHA),
        (prediction_path, PREDICTIONS_SHA),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Incorrect version of {path.name}.")

    with TemporaryDirectory(prefix="m5-audio-") as tmp:
        repo = Path(tmp)
        remote = "https://github.com/thehimanshhu/Group-3-DS-and-AI-Lab-Project"
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "fetch", "-q", "--depth", "1", remote, SCORER], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=repo, check=True
        )
        sys.path.insert(0, str(repo))
        from src.metrics import eer
        from src.model import load_model
        from src.score import find_checkpoint, resolve_paths, run_config

        pred = pd.read_csv(prediction_path, keep_default_na=False)
        valid = pred.loc[pred["split"].eq("valid")]
        _, threshold = eer(valid["label"].eq("fake"), valid["p_fake"])
        test = pred.loc[pred["split"].str.startswith("test_")].copy()
        manifest = pd.read_csv(
            manifest_path, usecols=KEY + ["path", "label"], keep_default_na=False
        )
        manifest = manifest.loc[manifest["split"].str.startswith("test_")]
        rows = test.merge(
            manifest,
            how="outer",
            on=KEY,
            suffixes=("", "_manifest"),
            validate="one_to_one",
        )
        if len(rows) != 118_295 or not rows["label"].eq(rows["label_manifest"]).all():
            raise ValueError("The manifest must match every scored test clip.")
        rows["wrong"] = rows["p_fake"].ge(threshold).ne(rows["label"].eq("fake"))
        rows = resolve_paths(rows, [data, english])

        cfg = run_config("F-03")
        checkpoint = find_checkpoint(
            "F-03", [find_input(inputs, "m4-training-notebook-3")]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(checkpoint, cfg, device)
        out = Path("/kaggle/working/audio_review")
        out.mkdir(parents=True, exist_ok=False)
        review_artifacts(select_cases(rows), out, model, device, cfg["data"], threshold)
        print("Prepared six private playback and saliency cases.", flush=True)

        sample_rate = cfg["data"]["sample_rate"]
        summaries = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for split, group in rows.groupby("split", sort=False):
                paths = group["path"].tolist()
                values = []
                for start in range(0, len(paths), 512):
                    values.extend(
                        pool.map(
                            lambda path: quiet_frame_db(
                                read_audio(path, sample_rate), sample_rate
                            ),
                            paths[start : start + 512],
                        )
                    )
                summaries.append(summarize_quiet(group.assign(quiet_frame_db=values)))
                print(f"{split}: measured {len(values):,} clips.", flush=True)
        pd.concat(summaries, ignore_index=True).to_csv(
            out / "quiet_frame_summary.csv", index=False
        )


if __name__ == "__main__":
    main()
