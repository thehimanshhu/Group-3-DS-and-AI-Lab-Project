"""Builds results/runs.csv from every results/<run_id>/run.json.

Each run writes its own run.json, so nobody edits a shared table. Run from the repository root:

    python -m src.collect_runs
"""
import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
COLUMNS = ["run_id", "owner", "param", "value", "epochs", "best_epoch", "train_eer", "best_valid_eer",
           "train_minutes", "git_commit", "kaggle_notebook"]


def collect(results=RESULTS):
    rows = [json.loads(p.read_text()) for p in sorted(results.glob("*/run.json"))]
    return pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("run_id", ignore_index=True)


if __name__ == "__main__":
    runs = collect()
    runs.to_csv(RESULTS / "runs.csv", index=False)
    print(f"{len(runs)} runs written to {RESULTS / 'runs.csv'}")
