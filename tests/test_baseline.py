"""CPU smoke tests for A-01 and A-02 using only generated audio.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_baseline_smoke.py' -v
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn

from src.baselines.aasist_l import AASISTL
from src.baselines.lfcc_lcnn import LFCCLCNN
from src.evaluate import score


SAMPLE_RATE = 16_000


class BaselineSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def check_training_step(self, model, weight, learning_rate):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        wave = torch.randn(2, 4 * SAMPLE_RATE)
        mask = torch.ones_like(wave, dtype=torch.long)
        wave[1, 3 * SAMPLE_RATE :] = 0
        mask[1, 3 * SAMPLE_RATE :] = 0
        labels = torch.tensor([0, 1])
        original_weight = weight.detach().clone()

        optimizer.zero_grad()
        logits = model(wave, mask)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all().item())

        loss = nn.functional.cross_entropy(logits, labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(weight.grad).all().item())

        optimizer.step()
        self.assertFalse(torch.equal(original_weight, weight.detach()))

    def test_a01_lfcc_lcnn_training_step(self):
        model = LFCCLCNN()
        with torch.inference_mode():
            features = model.lfcc(torch.randn(1, 4 * SAMPLE_RATE))
        self.assertEqual(tuple(features.shape), (1, 20, 401))
        self.assertTrue(torch.isfinite(features).all().item())
        self.check_training_step(model, model.classifier[-1].weight, 1e-3)

    def test_a02_aasist_l_training_step(self):
        model = AASISTL()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 85_306)
        self.check_training_step(model, model.network.out_layer.weight, 1e-4)

    def test_both_baselines_score_wav_files(self):
        with TemporaryDirectory() as directory:
            rows = pd.DataFrame({
                "clip_id": ["short", "long"],
                "path": ["short.wav", "long.wav"],
            })
            for filename, seconds in zip(rows["path"], (3, 6)):
                time = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
                waveform = 0.2 * np.sin(2 * np.pi * 440 * time)
                sf.write(Path(directory) / filename, waveform, SAMPLE_RATE)

            for model_type in (LFCCLCNN, AASISTL):
                with self.subTest(model=model_type.__name__):
                    result = score(
                        model_type(), rows, directory,
                        {"sample_rate": SAMPLE_RATE, "crop_seconds": 4.0},
                        "cpu",
                    )
                    pd.testing.assert_frame_equal(result[rows.columns], rows)
                    values = result[["p_fake", "p_fake_middle"]].to_numpy()
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(((values >= 0) & (values <= 1)).all())
                    self.assertAlmostEqual(values[0, 0], values[0, 1], places=6)
