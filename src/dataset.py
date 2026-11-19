"""Loading preprocessed clips for training and evaluation.

    rows = select_rows(manifest, cfg["data"], "train")
    ds = ClipDataset(rows, data_dir, cfg["data"], train=True, augment_cfg=cfg["augment"], seed=cfg["seed"])
    wave, mask, label, clip_id = ds[0]
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

LABELS = {"real": 0, "fake": 1}


def select_rows(manifest, data_cfg, split):
    """The manifest rows a run uses for one split.

    - data.tune_subset: true  → only the fixed 5,000-clip tuning subset of `train` (the sweeps)
    - data.exclude_language   → leave one language out of train and valid (cross-language runs)
    - split "train_eval"      → the fixed 1,000 training clips scored after each epoch (for the sweeps, 1,000 clips
                                of the tuning subset, so the training EER is measured on clips the run trains on)
    """
    if split == "train_eval":
        if data_cfg.get("tune_subset"):
            tune = manifest[(manifest.split == "train") & manifest.tune_subset.astype(bool)]
            rows = tune.sample(min(1000, len(tune)), random_state=42)
        else:
            rows = manifest[(manifest.split == "train") & manifest.train_eval.astype(bool)]
    else:
        rows = manifest[manifest.split == split]
        if split == "train" and data_cfg.get("tune_subset"):
            rows = rows[rows.tune_subset.astype(bool)]
    held_out = data_cfg.get("exclude_language")
    if held_out and split in ("train", "valid", "train_eval"):
        rows = rows[rows.language != held_out]
    return rows.reset_index(drop=True)


def any_augmentation(augment_cfg):
    return bool(augment_cfg) and any(isinstance(v, dict) and v.get("enabled") for v in augment_cfg.values())


class ClipDataset(Dataset):
    """Returns (waveform, attention mask, label, clip id) for each clip in `rows`.

    Training clips: a random crop of `data.crop_seconds`, then augmentation of that crop (if any effect is enabled).
    Other clips: the middle crop, never augmented. Every crop is scaled to zero mean and unit variance;
    shorter clips are zero-padded, and the mask marks which samples are audio (1) and which are padding (0).
    """

    def __init__(self, rows, data_dir, data_cfg, train, augment_cfg=None, seed=42):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.sr = data_cfg.get("sample_rate", 16000)
        self.samples = round(data_cfg.get("crop_seconds", 4.0) * self.sr)
        self.train = train
        self.augment_cfg = augment_cfg if train and any_augmentation(augment_cfg) else None
        self.seed = seed
        self.epoch = 0
        if self.augment_cfg is not None:
            from src import augment  # only needed when augmentation is on
            self._apply = augment.apply

    def set_epoch(self, epoch):
        """Different random crops and augmentations each epoch, the same on every re-run."""
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        audio, _ = sf.read(self.data_dir / row.path, dtype="float32")
        rng = np.random.default_rng([self.seed, self.epoch, i])
        if len(audio) > self.samples:
            start = int(rng.integers(0, len(audio) - self.samples + 1)) if self.train \
                else (len(audio) - self.samples) // 2
            audio = audio[start:start + self.samples]
        if self.augment_cfg is not None:  # on the crop; a speed change alters its length, so trim again
            audio = self._apply(audio, self.sr, self.augment_cfg, rng).astype(np.float32)[:self.samples]
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        wave, mask = np.zeros(self.samples, np.float32), np.zeros(self.samples, np.int64)
        wave[:len(audio)], mask[:len(audio)] = audio, 1
        return torch.from_numpy(wave), torch.from_numpy(mask), LABELS[row.label], row.clip_id


def read_manifest(data_dir, data_cfg):
    return pd.read_csv(Path(data_dir) / data_cfg.get("manifest", "manifests/manifest_m4.csv"), keep_default_na=False,
                       dtype={"speaker_id": str, "ref_speaker_id": str})
