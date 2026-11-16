"""Train one run from its config file.

    python -m src.train --config configs/T-03.yaml --data <folder with manifests/> --out <output folder> [--resume]

Writes into --out:
    config.yaml      the exact config of the run
    train_log.csv    epoch, step, loss, lr (one row per optimizer step)
    epoch_log.csv    epoch, train_loss, valid_loss, train_eer, valid_eer, minutes
    epoch_eval.csv   only with train.epoch_eval: EER, AUC and accuracy per split after each epoch
    valid_predictions.csv   p_fake for every valid clip at the best epoch
    best.pt          weights of the best epoch (lowest valid EER)
    last.pt          everything needed to resume; deleted when the run finishes
    run.json         one summary line for results/runs.csv

Settings come from the config (see configs/default.yaml). Settings that only some runs need, with defaults:
    data.tune_subset (false), data.exclude_language (none), train.epoch_eval (false), train.num_workers (2).
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ClipDataset, read_manifest, select_rows
from src.metrics import eer
from src.model import build_model

REPO = Path(__file__).resolve().parents[1]
EPOCH_EVAL_PER_TEST_SPLIT = 300


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dotted(cfg, key):
    """cfg["train"]["lr_encoder"] for key "train.lr_encoder"; None if missing."""
    for part in key.split("."):
        if not isinstance(cfg, dict) or part not in cfg:
            return None
        cfg = cfg[part]
    return cfg


def git_commit():
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def lr_factor(step, total, warmup, schedule):
    """Linear warmup, then cosine decay to 0 (or constant)."""
    if step < warmup:
        return (step + 1) / warmup
    if schedule != "cosine":
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def loader(rows, data_dir, cfg, train, batch_size):
    ds = ClipDataset(rows, data_dir, cfg["data"], train=train, augment_cfg=cfg.get("augment"), seed=cfg["seed"])
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=cfg["train"].get("num_workers", 2),
                      generator=torch.Generator().manual_seed(cfg["seed"]) if train else None, drop_last=False)


@torch.no_grad()
def predict(model, data, device, fp16, loss_fn=None):
    """(clip ids, labels, p_fake, mean loss) for every clip of a loader, in evaluation mode."""
    model.eval()
    ids, labels, scores, losses = [], [], [], []
    for wave, mask, label, clip_id in data:
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            logits = model(wave.to(device), mask.to(device)).float()
        if loss_fn is not None:
            losses.append(loss_fn(logits, label.to(device)).item() * len(label))
        scores += logits.softmax(dim=1)[:, 1].cpu().tolist()
        labels += label.tolist()
        ids += list(clip_id)
    mean_loss = sum(losses) / len(labels) if losses else float("nan")
    return ids, np.array(labels), np.array(scores), mean_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="folder that holds manifests/ and processed/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in --out")
    parser.add_argument("--limit", type=int, default=0, help="use at most this many clips per split (quick tests)")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="stop after this epoch and keep last.pt, to continue later with --resume "
                             "(e.g. to split a long run over two Kaggle sessions)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tr = cfg["train"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    set_seed(cfg["seed"])

    manifest = read_manifest(args.data, cfg["data"])
    rows = {split: select_rows(manifest, cfg["data"], split) for split in ("train", "valid", "train_eval")}
    if tr.get("epoch_eval"):  # fixed samples of every test split, scored after each epoch (only observed)
        for split in sorted(s for s in manifest.split.unique() if s.startswith("test_")):
            part = manifest[manifest.split == split]
            if "m4_eval" in part and part.m4_eval.astype(bool).any():
                part = part[part.m4_eval.astype(bool)]
            rows[split] = part.sample(min(EPOCH_EVAL_PER_TEST_SPLIT, len(part)), random_state=cfg["seed"])
    if args.limit:
        rows = {s: r.sample(min(args.limit, len(r)), random_state=cfg["seed"]).reset_index(drop=True)
                for s, r in rows.items()}
    print({s: len(r) for s, r in rows.items()})

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    fp16 = bool(tr.get("fp16", True)) and use_cuda
    model = build_model(cfg["model"])
    encoder_params, new_params = model.parameter_groups()
    model = model.to(device)
    if use_cuda and torch.cuda.device_count() > 1:  # Kaggle "GPU T4 x2"
        model = nn.DataParallel(model)
    core = model.module if isinstance(model, nn.DataParallel) else model

    counts = rows["train"].label.value_counts()
    if tr.get("class_weights", "from_train") == "from_train":  # computed from the train split only
        weights = [len(rows["train"]) / (2 * counts.get(name, 1)) for name in ("real", "fake")]
    else:
        weights = [1.0, 1.0]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    print(f"train clips {counts.to_dict()}, class weights (real, fake) {np.round(weights, 3).tolist()}")

    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": tr["lr_encoder"]},
                                   {"params": new_params, "lr": tr["lr_head"]}], weight_decay=tr.get("weight_decay", 0.0))
    accum = tr.get("grad_accum_steps", 1)
    train_data = loader(rows["train"], args.data, cfg, True, tr["batch_size"])
    eval_batch = 2 * tr["batch_size"]
    eval_data = {s: loader(r, args.data, cfg, False, eval_batch) for s, r in rows.items() if s != "train"}
    steps_per_epoch = math.ceil(len(train_data) / accum)
    total_steps = tr["epochs"] * steps_per_epoch
    warmup = round(tr.get("warmup_ratio", 0.0) * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup, tr.get("scheduler", "cosine")))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    state = {"epoch": 0, "step": 0, "best_eer": float("inf"), "best_epoch": 0, "bad_epochs": 0, "minutes": 0.0}
    if args.resume and (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        core.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        print(f"resumed after epoch {state['epoch']}")
    for name in ("train_log.csv", "epoch_log.csv", "epoch_eval.csv"):
        # keep only the rows of finished epochs (a stopped run may have logged part of the next one)
        log = out / name
        if log.exists():
            kept = pd.read_csv(log)
            kept = kept[kept.epoch <= state["epoch"]]
            if len(kept):
                kept.to_csv(log, index=False)
            else:
                log.unlink()

    patience = tr.get("early_stopping_patience")
    clip_norm = tr.get("grad_clip_norm")
    for epoch in range(state["epoch"] + 1, tr["epochs"] + 1):
        start = time.time()
        train_data.dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        step_rows, running = [], 0.0
        for i, (wave, mask, label, _) in enumerate(train_data, 1):
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                loss = loss_fn(model(wave.to(device), mask.to(device)), label.to(device)) / accum
            scaler.scale(loss).backward()
            running += loss.item()
            if i % accum == 0 or i == len(train_data):
                if clip_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                state["step"] += 1
                step_rows.append({"epoch": epoch, "step": state["step"], "loss": round(running, 5),
                                  "lr": scheduler.get_last_lr()[0]})
                running = 0.0
                if state["step"] % 50 == 0:
                    recent = np.mean([r["loss"] for r in step_rows[-50:]])
                    print(f"epoch {epoch} step {state['step']}/{total_steps} loss {recent:.3f} "
                          f"({(time.time() - start) / 60:.1f} min)", flush=True)
        pd.DataFrame(step_rows).to_csv(out / "train_log.csv", mode="a", index=False,
                                       header=not (out / "train_log.csv").exists())

        ids, labels, p_fake, valid_loss = predict(model, eval_data["valid"], device, fp16, loss_fn)
        valid_eer, threshold = eer(labels, p_fake)
        _, train_labels, train_scores, _ = predict(model, eval_data["train_eval"], device, fp16)
        train_eer, _ = eer(train_labels, train_scores)
        minutes = (time.time() - start) / 60
        state["minutes"] += minutes
        row = {"epoch": epoch, "train_loss": round(float(np.mean([r["loss"] for r in step_rows])), 5),
               "valid_loss": round(valid_loss, 5), "train_eer": round(train_eer, 5), "valid_eer": round(valid_eer, 5),
               "minutes": round(minutes, 2)}
        pd.DataFrame([row]).to_csv(out / "epoch_log.csv", mode="a", index=False,
                                   header=not (out / "epoch_log.csv").exists())
        print(f"epoch {epoch}: train EER {train_eer:.3f}, valid EER {valid_eer:.3f}, {minutes:.1f} min", flush=True)

        if tr.get("epoch_eval"):
            eval_rows = []
            for split in [s for s in eval_data if s not in ("valid",)] + ["valid"]:
                if split == "valid":
                    y, p = labels, p_fake
                elif split == "train_eval":
                    y, p = train_labels, train_scores
                else:
                    _, y, p, _ = predict(model, eval_data[split], device, fp16)
                if len(set(y)) < 2:
                    continue
                eval_rows.append({"epoch": epoch, "split": "train" if split == "train_eval" else split,
                                  "eer": round(eer(y, p)[0], 5), "auc": round(roc_auc_score(y, p), 5),
                                  "accuracy": round(float(((p >= threshold) == (y == 1)).mean()), 5)})
            pd.DataFrame(eval_rows).to_csv(out / "epoch_eval.csv", mode="a", index=False,
                                           header=not (out / "epoch_eval.csv").exists())

        if valid_eer < state["best_eer"]:
            state.update(best_eer=valid_eer, best_epoch=epoch, best_train_eer=train_eer, bad_epochs=0)
            torch.save({"model": core.state_dict(), "epoch": epoch, "valid_eer": valid_eer, "config": cfg},
                       out / "best.pt")
            valid_rows = rows["valid"].set_index("clip_id").loc[ids]
            valid_rows.assign(split="valid", p_fake=np.round(p_fake, 5)).reset_index()[
                ["clip_id", "split", "language", "dataset", "generator", "label", "p_fake"]
            ].to_csv(out / "valid_predictions.csv", index=False)
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch
        torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state},
                   out / "last.pt")
        if patience and state["bad_epochs"] >= patience:
            print(f"early stopping: valid EER hasn't improved for {patience} epochs")
            break
        if args.stop_after and epoch == args.stop_after and epoch < tr["epochs"]:
            print(f"stopped after epoch {epoch}; continue with --resume")
            return

    (out / "last.pt").unlink(missing_ok=True)
    experiment = cfg.get("experiment", {})
    param = experiment.get("param")
    run = {"run_id": cfg["run_id"], "owner": experiment.get("owner"), "param": param,
           "value": experiment.get("value", dotted(cfg, param) if param else None),
           "epochs": state["epoch"], "best_epoch": state["best_epoch"],
           "train_eer": round(state.get("best_train_eer", float("nan")), 5),
           "best_valid_eer": round(state["best_eer"], 5), "train_minutes": round(state["minutes"], 1),
           "git_commit": git_commit(), "kaggle_notebook": ""}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run))


if __name__ == "__main__":
    main()
