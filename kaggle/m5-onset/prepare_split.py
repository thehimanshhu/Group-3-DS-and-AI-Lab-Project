"""Stage one split for the F-03 onset check without copying validation audio.

From the repo root in the private scoring environment:

    PYTHONPATH=. python kaggle/m5-onset/prepare_split.py \
        --data <manifest-root> <extra-audio-root> --split test_itw --stage <new-scratch-dir>
    python -m src.score F-03 --full --data <new-scratch-dir> \
        --checkpoints <checkpoint-root> --out <separate-score-dir>

Use one scratch directory and one score directory per split. Keep the staging
directory outside the Kaggle output directory: it contains private audio.
With no staging flags on Kaggle, run the five splits and export scores only:
    kaggle kernels push -p kaggle/m5-onset --accelerator NvidiaTeslaT4
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import soundfile as sf

TEST_SPLITS = (
    "test_indomain",
    "test_unseen_gen",
    "test_phone",
    "test_itw",
    "test_asvspoof",
)
MANIFEST_COLUMNS = [
    "clip_id",
    "split",
    "language",
    "dataset",
    "generator",
    "label",
    "path",
]
# The XLS-R 300M convolutional stack needs 400 real samples for one feature frame.
MIN_REMAINING_SAMPLES = 400
INTACT_COMMIT = "1756e7e4b9a1f0320e664c34f0901648f5fc0880"
MANIFEST_SHA256 = "35826f269b44f83afd914342c16190f2f69c75f0127af8fbf2475bd9995412c8"


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


def stage_split(data_roots, split, stage):
    from src.dataset import read_manifest
    from src.score import resolve_paths, run_config

    cfg = run_config("F-03")["data"]
    manifest_path = Path(cfg["manifest"])
    sample_rate = cfg["sample_rate"]
    manifest = read_manifest(data_roots[0], cfg)
    selected = manifest.loc[manifest["split"].isin(["valid", split]), MANIFEST_COLUMNS]
    if (
        set(selected["split"]) != {"valid", split}
        or selected.duplicated(["split", "clip_id"]).any()
    ):
        raise ValueError(
            "Expected validation and test rows with unique split/clip IDs."
        )
    rows = resolve_paths(selected.reset_index(drop=True), data_roots)

    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "audio").mkdir()
    kept, excluded = [], []
    for index, row in enumerate(rows.to_dict("records")):
        if row["split"] == "valid":
            kept.append(row)
            continue
        with sf.SoundFile(row["path"]) as source:
            if (source.samplerate, source.channels, source.subtype) != (
                sample_rate,
                1,
                "PCM_16",
            ):
                raise ValueError(f"Expected {sample_rate} Hz mono PCM16: {row['path']}")
            remaining = source.frames - sample_rate
            if remaining < MIN_REMAINING_SAMPLES:
                excluded.append(
                    {
                        **{k: row[k] for k in ("split", "clip_id", "label")},
                        "original_samples": source.frames,
                        "remaining_samples": remaining,
                    }
                )
                continue
            source.seek(sample_rate)  # Exact 1 s cut; no resampling or renormalization.
            audio = source.read(dtype="int16")
        target = stage / "audio" / f"{index:06}.wav"
        sf.write(target, audio, sample_rate, subtype="PCM_16")
        kept.append({**row, "path": str(target)})

    staged = pd.DataFrame(kept, columns=MANIFEST_COLUMNS)
    if set(staged.loc[staged["split"].eq(split), "label"]) != {"real", "fake"}:
        raise ValueError("Both test labels must remain after trimming to compute EER.")
    (stage / manifest_path.parent).mkdir(parents=True, exist_ok=True)
    staged.to_csv(stage / manifest_path, index=False)
    pd.DataFrame(
        excluded,
        columns=["split", "clip_id", "label", "original_samples", "remaining_samples"],
    ).to_csv(stage / "excluded.csv", index=False)
    print(
        f"{split}: staged {staged['split'].eq(split).sum():,} test clips; excluded {len(excluded):,}."
    )


def score_full_on_kaggle():
    inputs = Path("/kaggle/input")
    data = find_input(inputs, "dsai-group3-m4-data")
    english = find_input(inputs, "dsai-group3-m5-full-tests-en")
    manifest = data / "manifests/manifest_m4.csv"
    if (
        not english.is_dir()
        or hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA256
    ):
        raise ValueError(
            "Expected the same version 4 manifest and English audio mount as the intact run."
        )

    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError("The intact run used two T4s; request NvidiaTeslaT4.")
    print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(2)]}", flush=True)
    out = Path("/kaggle/working/onset")
    with TemporaryDirectory(prefix="m5-onset-") as scratch:
        scratch = Path(scratch)
        repo = scratch / "repo"
        repo.mkdir()
        remote = "https://github.com/thehimanshhu/Group-3-DS-and-AI-Lab-Project"
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "fetch", "-q", "--depth", "1", remote, INTACT_COMMIT],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=repo, check=True
        )
        sys.path.insert(0, str(repo))

        from src.score import find_checkpoint

        checkpoint = find_checkpoint("F-03", [inputs])
        print(
            f"Using scorer {INTACT_COMMIT}, manifest {MANIFEST_SHA256}, checkpoint {checkpoint}",
            flush=True,
        )
        out.mkdir(parents=True, exist_ok=False)
        for split in TEST_SPLITS:
            with TemporaryDirectory(prefix=f"{split}-", dir=scratch) as split_dir:
                # The largest trimmed split is about 3.5 GB before filesystem overhead.
                free = shutil.disk_usage(split_dir).free
                print(f"{split}: {free / 1e9:.2f} GB scratch free", flush=True)
                if free < 5_000_000_000:
                    raise OSError("Need 5 GB of private scratch for one split.")
                stage = Path(split_dir) / "stage"
                scores = Path(split_dir) / "scores"
                stage_split([data, english], split, stage)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.score",
                        "F-03",
                        "--full",
                        "--data",
                        str(stage),
                        "--checkpoints",
                        str(checkpoint.parent.parent),
                        "--out",
                        str(scores),
                    ],
                    cwd=repo,
                    check=True,
                )
                shutil.copyfile(
                    scores / "F-03/predictions_full.csv.gz",
                    out / f"{split}_predictions.csv.gz",
                )
                shutil.copyfile(stage / "excluded.csv", out / f"{split}_excluded.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Source roots; the first contains the manifest.",
    )
    parser.add_argument("--split", choices=TEST_SPLITS, required=True)
    parser.add_argument(
        "--stage",
        type=Path,
        required=True,
        help="New private scratch directory for this split.",
    )
    args = parser.parse_args()
    stage_split(args.data, args.split, args.stage)


if __name__ == "__main__":
    local = any(
        a in ("-h", "--help") or a == "--stage" or a.startswith("--stage=")
        for a in sys.argv[1:]
    )
    if Path("/kaggle/input").is_dir() and not local:
        score_full_on_kaggle()
    else:
        main()
