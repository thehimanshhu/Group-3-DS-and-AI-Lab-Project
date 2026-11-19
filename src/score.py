"""Score trained runs on valid and the Milestone-4 test sample (A-03).

    python -m src.score --data <folder with manifests/> --checkpoints <folder(s) holding <run_id>/best.pt> \
        --out <output folder> F-01 D-01 C-hi ...

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
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="score at most this many clips per split (quick tests)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for run_id in args.runs:
        start = time.time()
        cfg = run_config(run_id)
        manifest = read_manifest(args.data, cfg["data"])
        rows = manifest[(manifest.split == "valid") | manifest.m4_eval.astype(str).eq("True")]
        if args.limit:
            rows = rows.groupby("split").head(args.limit)
        model = load_model(find_checkpoint(run_id, args.checkpoints), cfg, device)
        predictions = score(model, rows.reset_index(drop=True), args.data, cfg["data"], device)[COLUMNS]
        predictions[["p_fake", "p_fake_middle"]] = predictions[["p_fake", "p_fake_middle"]].round(5)
        folder = Path(args.out) / run_id
        folder.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(folder / "predictions.csv", index=False)
        results = summarise(run_id, predictions, cfg) | {"git_commit": commit}
        (folder / "results.json").write_text(json.dumps(results, indent=1))
        print(f"{run_id}: {len(predictions):,} clips in {(time.time() - start) / 60:.1f} min | EER "
              + ", ".join(f"{s} {v * 100:.1f}%" for s, v in results["eer"].items()), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
