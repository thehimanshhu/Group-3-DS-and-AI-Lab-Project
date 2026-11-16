"""Training-script checks on synthetic clips; no corpus, download or GPU needed.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_train.py' -v
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import ClipDataset, select_rows  # noqa: E402
from src.model import build_model, load_model  # noqa: E402

TINY = {"name": "tiny", "freeze_feature_encoder": True, "freeze_transformer_layers": 0, "pooling": "mean",
        "dropout": 0.1}


def make_data(folder, per_split=16):
    """Sine-wave clips of 1-6 s in train / valid / test_itw, with the manifest columns the code uses."""
    rng = np.random.default_rng(0)
    rows = []
    for split in ("train", "valid", "test_itw"):
        for i in range(per_split):
            label = "fake" if i % 2 else "real"
            language = ["hi", "ta", "en"][i % 3]
            seconds = rng.uniform(1, 6)
            t = np.arange(int(16000 * seconds)) / 16000
            wave = 0.1 * np.sin(2 * np.pi * (300 if label == "real" else 900) * t).astype(np.float32)
            path = Path("processed") / split / f"{split}_{i}.wav"
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(folder / path, wave, 16000, subtype="PCM_16")
            rows.append({"clip_id": f"{split}_{i}", "path": str(path), "dataset": "synthetic", "language": language,
                         "label": label, "generator": "none", "speaker_id": str(i), "ref_speaker_id": str(i),
                         "split": split, "tune_subset": split == "train" and i < 8,
                         "train_eval": split == "train" and i % 4 in (0, 1), "m4_eval": split == "test_itw"})
    (folder / "manifests").mkdir()
    pd.DataFrame(rows).to_csv(folder / "manifests" / "manifest_m4.csv", index=False)
    return pd.DataFrame(rows)


def make_config(folder, **train):
    cfg = {"run_id": "test", "seed": 42, "experiment": {"owner": "Tarika", "param": "train.lr_encoder", "value": 1e-4},
           "data": {"manifest": "manifests/manifest_m4.csv", "sample_rate": 16000, "crop_seconds": 2.0},
           "model": dict(TINY),
           "train": {"epochs": 2, "batch_size": 4, "grad_accum_steps": 1, "lr_encoder": 1e-4, "lr_head": 1e-3,
                     "weight_decay": 0.01, "scheduler": "cosine", "warmup_ratio": 0.1, "grad_clip_norm": 1.0,
                     "fp16": True, "class_weights": "from_train", "early_stopping_patience": 2, "num_workers": 0,
                     **train},
           "augment": {"noise": {"enabled": False}}}
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def train(config, data, out, *extra):
    return subprocess.run([sys.executable, "-m", "src.train", "--config", str(config), "--data", str(data),
                           "--out", str(out), *extra], cwd=ROOT, capture_output=True, text=True)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.manifest = make_data(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_crop_pad_and_mask(self):
        rows = select_rows(self.manifest, {}, "valid")
        ds = ClipDataset(rows, self.data, {"crop_seconds": 2.0}, train=False)
        for i in range(len(ds)):
            wave, mask, label, _ = ds[i]
            self.assertEqual(wave.shape, (32000,))
            audio = int(min(rows.iloc[i].pipe(lambda r: sf.info(self.data / r.path).frames), 32000))
            self.assertEqual(int(mask.sum()), audio)          # 1 for audio, 0 for padding
            self.assertTrue(torch.all(wave[audio:] == 0))       # padding is zeros
            self.assertIn(label, (0, 1))

    def test_evaluation_crop_is_the_same_every_time(self):
        rows = select_rows(self.manifest, {}, "valid")
        ds = ClipDataset(rows, self.data, {"crop_seconds": 2.0}, train=False)
        self.assertTrue(torch.equal(ds[0][0], ds[0][0]))

    def test_row_selection(self):
        self.assertEqual(len(select_rows(self.manifest, {"tune_subset": True}, "train")), 8)
        self.assertEqual(len(select_rows(self.manifest, {}, "train_eval")), 8)
        held_out = select_rows(self.manifest, {"exclude_language": "hi"}, "train")
        self.assertNotIn("hi", set(held_out.language))
        tune_eval = select_rows(self.manifest, {"tune_subset": True}, "train_eval")
        self.assertTrue(set(tune_eval.clip_id) <= set(select_rows(self.manifest, {"tune_subset": True}, "train").clip_id))


class ModelTests(unittest.TestCase):
    def test_shapes_and_padding_do_not_change_the_score(self):
        torch.manual_seed(0)
        for pooling in ("mean", "attentive"):
            model = build_model({**TINY, "pooling": pooling}).eval()
            wave = torch.randn(1, 16000)
            mask = torch.ones(1, 16000, dtype=torch.long)
            padded = torch.cat([wave, torch.zeros(1, 16000)], dim=1)
            padded_mask = torch.cat([mask, torch.zeros(1, 16000, dtype=torch.long)], dim=1)
            with torch.no_grad():
                logits = model(wave, mask)
                logits_padded = model(padded, padded_mask)
            with self.subTest(pooling=pooling):
                self.assertEqual(tuple(logits.shape), (1, 2))
                self.assertTrue(torch.allclose(logits, logits_padded, atol=1e-4))

    def test_frozen_layers(self):
        model = build_model({**TINY, "freeze_transformer_layers": 1})
        first, second = model.encoder.encoder.layers
        self.assertFalse(any(p.requires_grad for p in first.parameters()))
        self.assertTrue(all(p.requires_grad for p in second.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.encoder.feature_extractor.parameters()))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.data.mkdir()
        make_data(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_run_writes_every_output(self):
        out = Path(self.tmp.name) / "out"
        result = train(make_config(Path(self.tmp.name), epoch_eval=True), self.data, out)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        for name in ("config.yaml", "train_log.csv", "epoch_log.csv", "epoch_eval.csv", "valid_predictions.csv",
                     "best.pt", "run.json"):
            self.assertTrue((out / name).exists(), name)
        self.assertFalse((out / "last.pt").exists())
        epochs = pd.read_csv(out / "epoch_log.csv")
        self.assertEqual(list(epochs.columns), ["epoch", "train_loss", "valid_loss", "train_eer", "valid_eer", "minutes"])
        self.assertEqual(list(epochs.epoch), [1, 2])
        self.assertEqual(set(pd.read_csv(out / "epoch_eval.csv").split), {"train", "valid", "test_itw"})
        run = json.loads((out / "run.json").read_text())
        self.assertEqual(run["param"], "train.lr_encoder")
        self.assertEqual(run["epochs"], 2)
        self.assertIn(run["best_epoch"], (1, 2))
        model = load_model(out / "best.pt", yaml.safe_load((out / "config.yaml").read_text()), "cpu")
        self.assertEqual(tuple(model(torch.randn(2, 32000), torch.ones(2, 32000, dtype=torch.long)).shape), (2, 2))

    def test_stop_and_resume(self):
        out = Path(self.tmp.name) / "out"
        config = make_config(Path(self.tmp.name), epochs=3, early_stopping_patience=None)
        first = train(config, self.data, out, "--stop-after", "1")
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        self.assertTrue((out / "last.pt").exists())
        self.assertFalse((out / "run.json").exists())
        resumed = train(config, self.data, out, "--resume")
        self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:])
        self.assertIn("resumed after epoch 1", resumed.stdout)
        self.assertEqual(list(pd.read_csv(out / "epoch_log.csv").epoch), [1, 2, 3])
        steps = pd.read_csv(out / "train_log.csv").step
        self.assertEqual(list(steps), list(range(1, len(steps) + 1)))  # no step logged twice or skipped
        self.assertEqual(json.loads((out / "run.json").read_text())["epochs"], 3)
        self.assertFalse((out / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
