"""Check evaluation against Milestone 3 predictions, audio, and weights.

Set M3_DATA_DIR and M3_CHECKPOINT to enable the real-audio checks.
"""

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (
    compare_source_thresholds,
    evaluate_generators,
    evaluate_languages,
    evaluate_splits,
)
from src.metrics import eer


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "notebooks/Milestone-3/run_outputs/predictions.csv"
SAMPLE_RATE = 16_000


class EvaluationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(PREDICTIONS)

    def test_eer_on_perfect_scores_and_saved_validation_predictions(self):
        self.assertEqual(eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), (0.0, 0.8))
        valid = self.predictions.loc[self.predictions["split"].eq("valid")]
        error, threshold = eer(valid["label"].eq("fake"), valid["p_fake"])
        self.assertEqual(len(valid), 397)
        self.assertAlmostEqual(error, 0.244327, places=6)
        self.assertAlmostEqual(threshold, 0.31553, places=5)

    def test_asvspoof_thresholds_and_confusion_counts(self):
        summary = evaluate_splits(self.predictions).set_index("split")
        counts = summary.loc["test_asvspoof", [
            "real_as_real", "real_as_fake", "fake_as_real", "fake_as_fake",
        ]]
        self.assertEqual(counts.tolist(), [0, 48, 0, 48])
        self.assertEqual(summary["threshold"].nunique(), 1)

        comparison = compare_source_thresholds(
            self.predictions, "asvspoof2019_la", "test_asvspoof"
        ).set_index("threshold_type")
        source = comparison.loc["source"]
        self.assertAlmostEqual(source["threshold"], 0.37740, places=5)
        self.assertEqual(source[["real_as_fake", "fake_as_real"]].tolist(), [14, 19])

    def test_generator_and_english_reports(self):
        generators = evaluate_generators(self.predictions)
        self.assertEqual(generators["detected"].sum(), 150)
        self.assertEqual(generators["fake_clips"].sum(), 270)

        languages = evaluate_languages(self.predictions)
        english = languages.loc[languages["language"].eq("en")]
        self.assertEqual(set(english["split"]), {"test_asvspoof", "test_itw"})


class Milestone3AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = os.environ.get("M3_DATA_DIR")
        checkpoint = os.environ.get("M3_CHECKPOINT")
        if not data_dir and not checkpoint:
            raise unittest.SkipTest("Set M3_DATA_DIR and M3_CHECKPOINT for audio tests.")
        if not data_dir or not checkpoint:
            raise ValueError("Set both M3_DATA_DIR and M3_CHECKPOINT.")

        cls.data_dir = Path(data_dir)
        manifest_path = cls.data_dir / "manifests" / "manifest_subset.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Milestone 3 manifest not found: {manifest_path}")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Milestone 3 checkpoint not found: {checkpoint}")

        import soundfile as sf
        import torch
        from torch import nn
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        valid = manifest.loc[manifest["split"].eq("valid")]
        for index, row in valid.iterrows():
            wav_path = cls.data_dir / row["path"]
            info = sf.info(wav_path)
            if info.channels == 1 and info.samplerate == SAMPLE_RATE and 4 < info.duration <= 8:
                cls.rows = valid.loc[[index]]
                cls.audio, _ = sf.read(wav_path, dtype="float32")
                break
        else:
            raise ValueError("No 16 kHz mono validation clip longer than 4 s and at most 8 s.")

        class Milestone3Detector(nn.Module):
            """Match the encoder and head in 02_pipeline_check.ipynb."""

            def __init__(self):
                super().__init__()
                config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
                self.encoder = Wav2Vec2Model(config)
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(config.hidden_size, 2))

            def forward(self, wave, mask):
                frames = self.encoder(wave, attention_mask=mask).last_hidden_state
                frame_mask = self.encoder._get_feature_vector_attention_mask(
                    frames.shape[1], mask
                )
                pooled = (frames * frame_mask[..., None]).sum(1)
                pooled = pooled / frame_mask.sum(1, keepdim=True)
                return self.head(pooled)

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = Milestone3Detector()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cls.model.load_state_dict(weights)
        cls.model.to(cls.device).eval()

    def test_score_returns_both_probabilities_on_m3_audio(self):
        from src.evaluate import score

        result = score(
            self.model, self.rows, self.data_dir,
            {"sample_rate": SAMPLE_RATE, "crop_seconds": 4.0}, self.device,
        )
        pd.testing.assert_frame_equal(result[self.rows.columns], self.rows)
        probabilities = result[["p_fake", "p_fake_middle"]].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_saliency_is_finite_and_aligned_with_audio(self):
        from src.explain import saliency

        values = saliency(self.model, self.audio, self.device)
        self.assertEqual(values.shape, self.audio.shape)
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values.max(), 0)
