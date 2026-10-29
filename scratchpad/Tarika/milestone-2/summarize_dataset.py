"""Summarise and check the generated (fake) speech dataset.

Reads every manifest of the results repository on Hugging Face (or a local copy of it),
checks the data for missing, failed, duplicate and inconsistent clips, and writes summary
tables (CSV + Markdown) and figures (PNG) for the Milestone-2 report.

Usage:
    python summarize_dataset.py                       # read the manifests from Hugging Face
    python summarize_dataset.py --local-dir <folder>  # folder that contains manifests/
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HF_REPO = "TarikaHedaoo/dsai-group3-fake-speech"
GENERATORS = ["indic-parler-tts", "indicf5", "mms-tts"]
GENERATOR_NAMES = {"indic-parler-tts": "Indic Parler-TTS", "indicf5": "IndicF5", "mms-tts": "MMS-TTS"}
LANGUAGES = ["hi", "ta", "te"]
LANGUAGE_NAMES = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu"}
REQUIRED = ["clip_id", "sent_id", "lang", "generator", "text", "path", "sha1", "sample_rate",
            "duration_s", "voice_id", "voice_gender", "seed", "model_id", "label", "split"]

# Colours: one fixed hue per generator, checked for colour-blind separation.
COLORS = {"indic-parler-tts": "#2a78d6", "indicf5": "#eb6834", "mms-tts": "#1baf7a"}
GENDER_COLORS = {"f": "#2a78d6", "m": "#eb6834", "unknown": "#b4b2a9"}  # MMS-TTS: one voice, gender not stated
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_manifests(local_dir):
    if local_dir is None:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["manifests/*"])
    rows = []
    for path in sorted(glob.glob(f"{local_dir}/manifests/*/*/*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    return pd.DataFrame(rows)


def run_checks(df, sentences):
    """Return a list of (check, result, ok) rows."""
    checks = []

    def add(name, value, ok):
        checks.append({"check": name, "result": value, "ok": "yes" if ok else "NO"})

    expected = set(sentences["clip_id"])
    present = set(df["clip_id"])
    add("Expected clips (sentences.csv)", len(expected), True)
    add("Clips in manifests", len(df), len(df) == len(expected))
    add("Missing clips", len(expected - present), not (expected - present))
    add("Unexpected clips", len(present - expected), not (present - expected))
    add("Duplicate clip_id", int(df["clip_id"].duplicated().sum()), not df["clip_id"].duplicated().any())

    ok = df[df["error"].isna()]
    add("Failed clips (no audio)", len(df) - len(ok), len(ok) == len(df))
    add("Duplicate audio files (same sha1)", int(ok["sha1"].duplicated().sum()), not ok["sha1"].duplicated().any())

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    add("Missing manifest columns", ", ".join(missing_cols) or "none", not missing_cols)
    n_null = int(ok[[c for c in REQUIRED if c in df.columns]].isna().sum().sum())
    add("Empty values in required columns", n_null, n_null == 0)

    merged = df.merge(sentences[["clip_id", "text", "lang", "generator", "sent_id"]], on="clip_id",
                      suffixes=("", "_expected"))
    mismatched = merged[(merged["text"] != merged["text_expected"]) | (merged["lang"] != merged["lang_expected"])
                        | (merged["generator"] != merged["generator_expected"])
                        | (merged["sent_id"] != merged["sent_id_expected"])]
    add("Clips whose text/language/generator differ from sentences.csv", len(mismatched), mismatched.empty)

    per_sentence = df.groupby("sent_id")["generator"].nunique()
    add("Sentences spoken by all 3 generators", f"{int((per_sentence == 3).sum())} of {len(per_sentence)}",
        bool((per_sentence == 3).all()))

    rates = df.groupby("generator")["sample_rate"].nunique()
    add("One sample rate per generator",
        ", ".join(f"{g}: {', '.join(str(int(r)) for r in sorted(df[df.generator == g].sample_rate.unique()))} Hz"
                  for g in GENERATORS), bool((rates == 1).all()))
    add("Labels", ", ".join(sorted(df["label"].unique())), set(df["label"]) == {"fake"})
    add("Splits", ", ".join(sorted(df["split"].unique())), set(df["split"]) == {"test"})

    add("Clips flagged silent", int(df["flag_silent"].sum()), True)
    add("Clips flagged too short", int(df["flag_short"].sum()), True)
    add("Clips with > 1% clipped samples", int((df["clipped_frac"] > 0.01).sum()), True)
    return pd.DataFrame(checks)


def overview_table(df):
    rows = []
    for gen in GENERATORS:
        for lang in LANGUAGES + ["all"]:
            part = df[df.generator == gen] if lang == "all" else df[(df.generator == gen) & (df.lang == lang)]
            ok = part[part["error"].isna()]
            rows.append({
                "generator": GENERATOR_NAMES[gen], "language": LANGUAGE_NAMES.get(lang, "All"),
                "clips": len(part), "failed": len(part) - len(ok),
                "flagged": int((part["flag_silent"] | part["flag_short"]).sum()),
                "hours": round(ok["duration_s"].sum() / 3600, 2),
                "dur_min_s": round(ok["duration_s"].min(), 1), "dur_median_s": round(ok["duration_s"].median(), 1),
                "dur_max_s": round(ok["duration_s"].max(), 1),
                "sample_rate": int(ok["sample_rate"].iloc[0]),
                "voices": ok["voice_id"].nunique(),
                "female": int((ok["voice_gender"] == "f").sum()), "male": int((ok["voice_gender"] == "m").sum()),
                "gender_unknown": int((ok["voice_gender"] == "unknown").sum()),
            })
    total = df[df["error"].isna()]
    rows.append({"generator": "All", "language": "All", "clips": len(df), "failed": len(df) - len(total),
                 "flagged": int((df["flag_silent"] | df["flag_short"]).sum()),
                 "hours": round(total["duration_s"].sum() / 3600, 2),
                 "dur_min_s": round(total["duration_s"].min(), 1),
                 "dur_median_s": round(total["duration_s"].median(), 1),
                 "dur_max_s": round(total["duration_s"].max(), 1), "sample_rate": "",
                 "voices": total["voice_id"].nunique(),
                 "female": int((total["voice_gender"] == "f").sum()), "male": int((total["voice_gender"] == "m").sum()),
                 "gender_unknown": int((total["voice_gender"] == "unknown").sum())})
    return pd.DataFrame(rows)


def voices_table(df):
    table = (df.groupby(["generator", "lang", "voice_id", "voice_gender"]).size()
             .reset_index(name="clips").sort_values(["generator", "lang", "voice_id"]))
    table["generator"] = table["generator"].map(GENERATOR_NAMES)
    table["lang"] = table["lang"].map(LANGUAGE_NAMES)
    return table


def text_table(sentences):
    one = sentences.drop_duplicates("sent_id")
    rows = []
    for lang in LANGUAGES:
        part = one[one.lang == lang]
        rows.append({"language": LANGUAGE_NAMES[lang], "sentences": len(part),
                     "kathbath_speakers": part["kb_speaker"].nunique(),
                     "words_min": part["n_words"].min(), "words_median": part["n_words"].median(),
                     "words_max": part["n_words"].max(),
                     "female_reader_share": round((part["kb_gender"] == "f").mean(), 2)})
    return pd.DataFrame(rows)


# ---------- figures ----------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def box_by_generator(df, column, ylabel, title, out_path):
    """One panel per language, one box per generator."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True, facecolor=SURFACE)
    for ax, lang in zip(axes, LANGUAGES):
        data = [df[(df.lang == lang) & (df.generator == g)][column].dropna() for g in GENERATORS]
        parts = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                           medianprops={"color": INK, "linewidth": 1.5},
                           whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED},
                           flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_MUTED,
                                       "markeredgecolor": "none", "alpha": 0.4})
        for box, gen in zip(parts["boxes"], GENERATORS):
            box.set_facecolor(COLORS[gen])
            box.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, 4), [GENERATOR_NAMES[g] for g in GENERATORS], fontsize=8.5, color=INK)
        ax.set_title(LANGUAGE_NAMES[lang], fontsize=10, color=INK)
        style_axes(ax)
    axes[0].set_ylabel(ylabel, color=INK, fontsize=9.5)
    fig.suptitle(title, color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def gender_figure(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    labels = []
    counts = {"f": [], "m": [], "unknown": []}
    for gen in GENERATORS:
        for lang in LANGUAGES:
            part = df[(df.generator == gen) & (df.lang == lang)]
            labels.append(f"{GENERATOR_NAMES[gen]} · {LANGUAGE_NAMES[lang]}")
            for gender in counts:
                counts[gender].append((part.voice_gender == gender).sum())
    y = np.arange(len(labels))[::-1]  # first generator at the top
    left = np.zeros(len(labels))
    names = {"f": "Female voice", "m": "Male voice", "unknown": "Not stated (MMS-TTS)"}
    for gender, values in counts.items():
        ax.barh(y, values, height=0.6, left=left, color=GENDER_COLORS[gender], label=names[gender],
                edgecolor=SURFACE, linewidth=2)
        left += np.array(values)
    ax.set_yticks(y, labels, fontsize=8.5, color=INK)
    ax.set_xlabel("Clips", color=INK, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=INK)
    ax.set_title("Voice gender per generator and language", color=INK, fontsize=11.5, loc="left")
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sentence_length_figure(sentences, out_path):
    one = sentences.drop_duplicates("sent_id")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True, facecolor=SURFACE)
    one_words = one["n_words"]
    bins = np.arange(one_words.min() - 0.5, one_words.max() + 1.5, 1)
    for ax, lang in zip(axes, LANGUAGES):
        ax.hist(one[one.lang == lang]["n_words"], bins=bins, color=COLORS["indic-parler-tts"],
                edgecolor=SURFACE, linewidth=2)
        ax.set_title(LANGUAGE_NAMES[lang], fontsize=10, color=INK)
        ax.set_xlabel("Words per sentence", color=INK, fontsize=9)
        ax.set_xticks(range(one_words.min(), one_words.max() + 1, 2))
        style_axes(ax)
    axes[0].set_ylabel("Sentences", color=INK, fontsize=9.5)
    fig.suptitle("Length of the input sentences (720 per language)", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(out_dir, checks, overview, voices, texts):
    def md(table):
        return table.to_markdown(index=False)

    lines = ["# Generated speech dataset: summary", "",
             f"Source: `{HF_REPO}` (manifests). Created by `summarize_dataset.py`.", "",
             "## Checks", "", md(checks), "",
             "## Clips per generator and language", "", md(overview), "",
             "## Input sentences", "", md(texts), "",
             "## Voices", "", md(voices), "",
             "## Figures", "",
             "- `fig_duration.png`: clip duration per generator and language",
             "- `fig_speaking_rate.png`: words per second per generator and language",
             "- `fig_loudness.png`: RMS level (dBFS) per generator and language",
             "- `fig_gender.png`: voice gender per generator and language",
             "- `fig_sentence_length.png`: words per input sentence", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", help="folder that contains manifests/ (default: download from Hugging Face)")
    p.add_argument("--sentences", type=Path, default=HERE / "sentences.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "summary")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifests(args.local_dir)
    with open(args.sentences, encoding="utf-8") as f:
        sentences = pd.DataFrame(list(csv.DictReader(f)))
    sentences["n_words"] = sentences["n_words"].astype(int)

    df["words_per_s"] = df["n_words"] / df["duration_s"]
    df["rms_dbfs"] = 20 * np.log10(df["rms"].clip(lower=1e-6))

    checks = run_checks(df, sentences)
    overview = overview_table(df)
    voices = voices_table(df)
    texts = text_table(sentences)
    for name, table in [("checks", checks), ("overview", overview), ("voices", voices), ("sentences", texts)]:
        table.to_csv(args.out_dir / f"{name}.csv", index=False)

    ok = df[df["error"].isna()]
    box_by_generator(ok, "duration_s", "Duration (s)", "Clip duration", args.out_dir / "fig_duration.png")
    box_by_generator(ok, "words_per_s", "Words per second", "Speaking rate",
                     args.out_dir / "fig_speaking_rate.png")
    box_by_generator(ok, "rms_dbfs", "RMS level (dBFS)", "Loudness before normalisation",
                     args.out_dir / "fig_loudness.png")
    gender_figure(ok, args.out_dir / "fig_gender.png")
    sentence_length_figure(sentences, args.out_dir / "fig_sentence_length.png")
    write_markdown(args.out_dir, checks, overview, voices, texts)

    print(checks.to_string(index=False))
    print()
    print(overview[overview.language == "All"].to_string(index=False))
    print(f"\nTables and figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
