"""Score trained runs on valid and the Milestone-4 test sample (A-03).

    python -m src.score --data <folder with manifests/> [<more clip folders>] \
        --checkpoints <folder(s) holding <run_id>/best.pt> --out <output folder> [--full] F-01 D-01 C-hi ...

For each run, writes into <out>/<run_id>/:
    predictions.csv   clip_id, split, language, dataset, generator, label, p_fake, p_fake_middle
                      (p_fake: mean over 4 s windows every 2 s; p_fake_middle: the middle 4 s only)
    results.json      {"run_id", "git_commit", "threshold", "eer": {split: ...}, "auc": {...}, "accuracy": {...}}
                      The threshold is the EER point of valid. Cross-language runs (C-xx) also get
                      "held_out_language" and "eer_held_out": the EER on that language's test clips.

The config comes from results/<run_id>/config.yaml (or configs/<run_id>.yaml).
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.dataset import read_manifest
from src.evaluate import score
from src.metrics import eer
from src.model import load_model

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ["clip_id", "split", "language", "dataset", "generator", "label", "p_fake", "p_fake_middle"]


def run_config(run_id):
    for path in (REPO / "results" / run_id / "config.yaml", REPO / "configs" / f"{run_id}.yaml"):
        if path.exists():
            return yaml.safe_load(path.read_text())
    raise FileNotFoundError(f"No config for {run_id}")


def find_checkpoint(run_id, roots):
    for root in roots:
        found = sorted(Path(root).rglob(f"{run_id}/best.pt"))
        if found:
            return found[0]
    raise FileNotFoundError(f"No {run_id}/best.pt under {roots}")


def resolve_paths(rows, roots):
    """Each clip's absolute path, from the first data folder that has it."""
    roots = [Path(r).resolve() for r in roots]
    listed = {}  # (root, folder) -> file names; one listing per folder instead of one check per clip (slow on Kaggle)

    def has(root, p):
        folder = (root / p).parent
        if folder not in listed:
            listed[folder] = {e.name for e in os.scandir(folder)} if folder.is_dir() else set()
        return Path(p).name in listed[folder]

    paths = []
    for p in rows.path:
        root = next((r for r in roots if has(r, p)), None)
        if root is None:
            raise FileNotFoundError(f"{p} is in none of {[str(r) for r in roots]}")
        paths.append(str(root / p))
    return rows.assign(path=paths)


def summarise(run_id, predictions, cfg):
    valid = predictions[predictions.split == "valid"]
    _, threshold = eer(valid.label.eq("fake").astype(int), valid.p_fake)
    out = {"run_id": run_id, "threshold": round(threshold, 5), "eer": {}, "auc": {}, "accuracy": {}}
    for split, part in predictions.groupby("split", sort=False):
        y = part.label.eq("fake").astype(int)
        if y.nunique() < 2:
            continue
        out["eer"][split] = round(eer(y, part.p_fake)[0], 5)
        out["auc"][split] = round(roc_auc_score(y, part.p_fake), 5)
        out["accuracy"][split] = round(float(((part.p_fake >= threshold).astype(int) == y).mean()), 5)
    held_out = cfg["data"].get("exclude_language")
    if held_out:
        part = predictions[predictions.split.str.startswith("test_") & (predictions.language == held_out)]
        out["held_out_language"] = held_out
        out["eer_held_out"] = round(eer(part.label.eq("fake").astype(int), part.p_fake)[0], 5)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data", nargs="+", required=True,
                        help="folder(s) with the clips; the first one holds manifests/. A clip is read from the "
                             "first folder that has it (e.g. the main dataset plus the full English test sets)")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="score at most this many real and this many fake clips per split (quick tests)")
    parser.add_argument("--batch-size", type=int, default=64, help="audio windows per forward pass (split over all GPUs)")
    parser.add_argument("--fp32", action="store_true", help="score in fp32 (about 2.5x slower; fp16 moves only borderline scores)")
    parser.add_argument("--trim-start", type=float, default=0,
                        help="cut this many seconds from the start of each clip (onset check; clips that would keep "
                             "less than 1 s are scored whole); writes predictions_trim<N>s.csv / results_trim<N>s.json")
    parser.add_argument("--full", action="store_true",
                        help="score every test clip (Milestone 5), not just the Milestone-4 sample; writes "
                             "predictions_full.csv.gz and results_full.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for run_id in args.runs:
        start = time.time()
        cfg = run_config(run_id)
        manifest = read_manifest(args.data[0], cfg["data"])
        if args.full:
            test = manifest.split.str.startswith("test_")
            if cfg["data"].get("exclude_language"):  # cross-language runs: only the held-out language's test clips
                test &= manifest.language == cfg["data"]["exclude_language"]
            rows = manifest[(manifest.split == "valid") | test]
        else:
            rows = manifest[(manifest.split == "valid") | manifest.m4_eval.astype(str).eq("True")]
        if args.limit:
            rows = rows.groupby(["split", "label"]).head(args.limit)  # both classes in every split
        rows = resolve_paths(rows.reset_index(drop=True), args.data)
        model = load_model(find_checkpoint(run_id, args.checkpoints), cfg, device)
        predictions = score(model, rows, "/", cfg["data"], device, batch_size=args.batch_size,
                            fp16=not args.fp32, trim_start=args.trim_start)[COLUMNS]
        predictions[["p_fake", "p_fake_middle"]] = predictions[["p_fake", "p_fake_middle"]].round(5)
        folder = Path(args.out) / run_id
        folder.mkdir(parents=True, exist_ok=True)
        suffix = ("_full" if args.full else "") + (f"_trim{args.trim_start:g}s" if args.trim_start else "")
        predictions.to_csv(folder / f"predictions{suffix}.csv{'.gz' if args.full else ''}", index=False)
        results = summarise(run_id, predictions, cfg) | {"git_commit": commit, "clips": len(predictions)}
        (folder / f"results{suffix}.json").write_text(json.dumps(results, indent=1))
        print(f"{run_id}: {len(predictions):,} clips in {(time.time() - start) / 60:.1f} min | EER "
              + ", ".join(f"{s} {v * 100:.1f}%" for s, v in results["eer"].items()), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
